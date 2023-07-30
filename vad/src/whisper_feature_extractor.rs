use std::f32::consts::PI;
use std::include_bytes;
use std::sync::Arc;
use num::complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use lazy_static::lazy_static;
use npy::NpyData;

pub const AUDIO_BUFFER_SIZE: usize = 16000 * 30;
pub const N_FFT: usize = 400;
pub const HOP_SIZE: usize = 160;
pub const N_MELS: usize = 80;
pub const SPECTROGRAM_ROWS: usize = 201;
pub const N_FRAMES: usize = 3000;
pub const SPECTROGRAM_SIZE: usize = SPECTROGRAM_ROWS * N_FRAMES;
pub const FEATURES_SIZE: usize = N_MELS * N_FRAMES;

lazy_static! {
    static ref MELS: Vec<f32> = {
        let npy_bytes = include_bytes!("m80.npy");
        let npy_data: NpyData<f32> = NpyData::from_bytes(npy_bytes).unwrap();
        npy_data.to_vec().iter().map(|x| *x as f32).collect()
    };
}

pub struct WhisperFeatureExtractor {
    fft_plan: Arc<dyn RealToComplex<f32>>,
    window: [f32; N_FFT],
    spectrogram: Vec<f32>,
    features: Vec<f32>,
}

impl WhisperFeatureExtractor {
    pub fn new() -> WhisperFeatureExtractor {
        let mut planner = RealFftPlanner::new();
        let fft_plan = planner.plan_fft_forward(N_FFT);

        let window = (0..N_FFT)
            .map(|i| (i as f32 * 2.0 * PI) / (N_FFT as f32))
            .map(|i| (1.0 - i.cos()) / 2.0)
            .collect::<Vec<_>>()
            .as_slice()
            .try_into()
            .unwrap();

        let spectrogram = vec![0.0; SPECTROGRAM_SIZE];
        let features = vec![0.0; FEATURES_SIZE];

        WhisperFeatureExtractor {
            fft_plan,
            window,
            spectrogram,
            features,
        }
    }

    fn fft(&self, mut audio: [f32; N_FFT]) -> Vec<Complex<f32>> {
        for i in 0..N_FFT {
            audio[i] *= self.window[i];
        }
        let mut spectrum = self.fft_plan.make_output_vec();
        self.fft_plan.process(&mut audio, &mut spectrum).unwrap();
        spectrum
    }

    fn compute_spectrogram_column(&mut self, frame: [f32; N_FFT], column_idx: usize) {
        let fft = self.fft(frame.try_into().unwrap());
        let start = column_idx * SPECTROGRAM_ROWS;
        for i in 0..SPECTROGRAM_ROWS {
            self.spectrogram[start + i] = fft[i].norm_sqr();
        }
    }

    pub fn compute_spectrogram(&mut self, audio: &[f32; AUDIO_BUFFER_SIZE]) {
        let padding_len = N_FFT as i64 / 2;
        let mut spectrogram_column_idx = 0;
        for i in (-padding_len..(audio.len() as i64) - padding_len).step_by(HOP_SIZE) {
            let frame = if i < 0 {
                let n_pad = (-i) as usize;
                let mut padded_frame = [0.0; N_FFT];
                let pad_content = &audio[..(i + N_FFT as i64) as usize];
                padded_frame[n_pad..].copy_from_slice(pad_content);
                for i in 0..n_pad {
                    padded_frame[i] = padded_frame[2 * n_pad - i - 1];
                }
                padded_frame
            } else {
                let end_idx = std::cmp::min(i as usize + N_FFT, audio.len());
                let length = end_idx - i as usize;
                if length < N_FFT {
                    let mut padded_frame = [0.0; N_FFT];
                    let pad_content = &audio[i as usize..end_idx];
                    padded_frame[..length].copy_from_slice(pad_content);
                    for i in length..N_FFT {
                        padded_frame[i] = padded_frame[length * 2 - i - 1];
                    }
                    padded_frame
                } else {
                    audio[i as usize..end_idx].try_into().unwrap()
                }
            };
            self.compute_spectrogram_column(frame.try_into().unwrap(), spectrogram_column_idx);
            spectrogram_column_idx += 1;
        }
    }

    pub fn extract_features(&mut self, audio: &[f32; AUDIO_BUFFER_SIZE]) -> &[f32; FEATURES_SIZE] {
        assert_eq!(self.spectrogram.len(), SPECTROGRAM_SIZE);
        assert_eq!(self.features.len(), FEATURES_SIZE);

        self.compute_spectrogram(audio);

        for i in 0..N_FRAMES {
            for j in 0..N_MELS {
                let mut sum = 0.0;
                for k in 0..SPECTROGRAM_ROWS {
                    sum += self.spectrogram[i * SPECTROGRAM_ROWS + k] * MELS[j * SPECTROGRAM_ROWS + k];
                }
                self.features[j * N_FRAMES + i] = sum.max(1e-10).log10();
            }
        }

        let features_max = *self.features
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        for i in 0..FEATURES_SIZE {
            self.features[i] = (self.features[i].max(features_max - 8.0) + 4.0) / 4.0;
        }

        self.features
            .as_slice()
            .try_into()
            .unwrap()
    }
}