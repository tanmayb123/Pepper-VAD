use std::f32::consts::PI;
use std::include_bytes;
use std::sync::Arc;
use num::complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use lazy_static::lazy_static;
use npy::NpyData;

pub const N_FFT: usize = 400;
pub const HOP_SIZE: usize = 160;
pub const N_MELS: usize = 80;
pub const SPECTROGRAM_ROWS: usize = 201;
pub const MULTIPLIER: f32 = 10.0;
pub const TOP_DB: f32 = 80.0;

lazy_static! {
    static ref MELS: Vec<f32> = {
        let npy_bytes = include_bytes!("m80-speechbrain.npy");
        let npy_data: NpyData<f32> = NpyData::from_bytes(npy_bytes).unwrap();
        npy_data.to_vec().iter().map(|x| *x as f32).collect()
    };
}

pub struct SpeechBrainFeatureExtractor {
    fft_plan: Arc<dyn RealToComplex<f32>>,
    window: [f32; N_FFT],
}

impl SpeechBrainFeatureExtractor {
    pub fn new() -> SpeechBrainFeatureExtractor {
        let mut planner = RealFftPlanner::new();
        let fft_plan = planner.plan_fft_forward(N_FFT);

        let window = (0..N_FFT)
            .map(|i| (i as f32 * 2.0 * PI) / (N_FFT as f32))
            .map(|i| 0.54 - 0.46 * i.cos())
            .collect::<Vec<_>>()
            .as_slice()
            .try_into()
            .unwrap();

        SpeechBrainFeatureExtractor {
            fft_plan,
            window,
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

    fn compute_spectrogram_column(&self, frame: [f32; N_FFT], column_idx: usize, spectrogram: &mut [f32]) {
        let fft = self.fft(frame.try_into().unwrap());
        let start = column_idx * SPECTROGRAM_ROWS;
        for i in 0..SPECTROGRAM_ROWS {
            spectrogram[start + i] = fft[i].norm_sqr();
        }
    }

    pub fn compute_spectrogram(&self, audio: &[f32], spectrogram: &mut [f32]) {
        let padding_len = N_FFT as i64 / 2;
        let mut spectrogram_column_idx = 0;
        for i in (-padding_len..(audio.len() as i64) - padding_len).step_by(HOP_SIZE) {
            let frame = if i < 0 {
                let mut padded_frame = [0.0; N_FFT];
                let pad_content = &audio[..(i + N_FFT as i64) as usize];
                padded_frame[-i as usize..].copy_from_slice(pad_content);
                padded_frame
            } else {
                let end_idx = std::cmp::min(i as usize + N_FFT, audio.len());
                let length = end_idx - i as usize;
                if length < N_FFT {
                    let mut padded_frame = [0.0; N_FFT];
                    let pad_content = &audio[i as usize..end_idx];
                    padded_frame[..length].copy_from_slice(pad_content);
                    padded_frame
                } else {
                    audio[i as usize..end_idx].try_into().unwrap()
                }
            };
            self.compute_spectrogram_column(frame.try_into().unwrap(), spectrogram_column_idx, spectrogram);
            spectrogram_column_idx += 1;
        }
    }

    pub fn extract_features(&self, audio: &[f32]) -> Vec<f32> {
        let n_frames = audio.len() / HOP_SIZE + 1;
        let n_features = n_frames * N_MELS;

        let mut spectrogram = vec![0.0; n_frames * SPECTROGRAM_ROWS];
        self.compute_spectrogram(audio, &mut spectrogram);

        let mut features = vec![0.0; n_features];
        for i in 0..n_frames {
            for j in 0..N_MELS {
                let mut sum = 0.0;
                for k in 0..SPECTROGRAM_ROWS {
                    sum += spectrogram[i * SPECTROGRAM_ROWS + k] * MELS[j * SPECTROGRAM_ROWS + k];
                }
                features[i * N_MELS + j] = sum.max(1e-10).log10() * MULTIPLIER;
            }
        }

        let features_max = *features
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap() - TOP_DB;
        for i in 0..n_features {
            features[i] = features[i].max(features_max);
        }

        features
    }
}