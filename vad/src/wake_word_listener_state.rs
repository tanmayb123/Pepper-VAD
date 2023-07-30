use std::error::Error;
use lazy_static::lazy_static;
use samplerate::{ConverterType, Samplerate};
use crate::speech_brain_embedding_model::SpeechBrainEmbeddingModel;
use crate::speech_brain_feature_extractor::SpeechBrainFeatureExtractor;
use crate::whisper_feature_extractor::{AUDIO_BUFFER_SIZE, WhisperFeatureExtractor};
use crate::whisper_model::WhisperLikelihoodModel;
use crate::whisper_transcription::WhisperTranscriber;

pub const DESIRED_SAMPLE_RATE: usize = 16000;
pub const STREAM_FRAME_DURATION: usize = 1000;

pub const WAKE_WORD_WINDOW_DURATION: usize = 2;
pub const WAKE_WORD_ACTIVATION_THRESHOLD: f32 = 0.000001;
pub const SPEAKER_EMBEDDING_WINDOW_DURATION: usize = 2;
pub const SPEAKER_EMBEDDING_SIMILARITY_THRESHOLD: f32 = 0.7;

lazy_static! {
    static ref OPENAI_API_KEY: String = std::env::var("OPENAI_API_KEY").unwrap();
}

#[derive(Debug)]
pub enum WakeWordListenerMode {
    Background,
    Start,
    Endpoint,
}

pub struct WakeWordListenerState {
    recording_sample_rate: usize,
    recording_channels: usize,

    whisper_feature_extractor: WhisperFeatureExtractor,
    whisper_likelihood_model: WhisperLikelihoodModel,
    speech_brain_feature_extractor: SpeechBrainFeatureExtractor,
    speech_brain_embedding_model: SpeechBrainEmbeddingModel,
    whisper_transcriber: WhisperTranscriber,

    mode: WakeWordListenerMode,
    recorded_audio: Vec<f32>,
    speaker_embedding: Vec<f32>,
}

impl WakeWordListenerState {
    pub fn new(sample_rate: usize, channels: usize) -> Result<Self, Box<dyn Error>> {
        Ok(
            WakeWordListenerState{
                recording_sample_rate: sample_rate,
                recording_channels: channels,

                whisper_feature_extractor: WhisperFeatureExtractor::new(),
                whisper_likelihood_model: WhisperLikelihoodModel::new()?,
                speech_brain_feature_extractor: SpeechBrainFeatureExtractor::new(),
                speech_brain_embedding_model: SpeechBrainEmbeddingModel::new()?,
                whisper_transcriber: WhisperTranscriber::new(OPENAI_API_KEY.to_string()),

                mode: WakeWordListenerMode::Background,
                recorded_audio: Vec::new(),
                speaker_embedding: Vec::new(),
            }
        )
    }

    fn resample_audio(&self) -> Result<Vec<f32>, Box<dyn Error>> {
        let mono_audio = if self.recording_channels == 1 {
            self.recorded_audio.clone()
        } else {
            self.recorded_audio
                .chunks(self.recording_channels)
                .map(|chunk| {
                    chunk.iter().sum::<f32>() / self.recording_channels as f32
                })
                .collect()
        };
        if self.recording_sample_rate == DESIRED_SAMPLE_RATE {
            Ok(mono_audio)
        } else {
            let converter = Samplerate::new(
                ConverterType::SincBestQuality,
                self.recording_sample_rate as u32,
                DESIRED_SAMPLE_RATE as u32,
                1,
            )?;
            Ok(
                converter
                    .process_last(&mono_audio)?
            )
        }
    }

    pub fn process_audio(&mut self, buffer: Vec<f32>) -> Result<(), Box<dyn Error>> {
        match &self.mode {
            WakeWordListenerMode::Background => self.handle_background_buffer(buffer),
            WakeWordListenerMode::Start => self.handle_start_buffer(buffer),
            WakeWordListenerMode::Endpoint => self.handle_endpoint_buffer(buffer),
        }?;
        Ok(())
    }

    pub fn current_mode(&self) -> &WakeWordListenerMode {
        &self.mode
    }

    fn handle_background_buffer(&mut self, buffer: Vec<f32>) -> Result<(), Box<dyn Error>> {
        // We're in "wake word detection" mode, waiting for a start point.
        // Let's record some audio, get rid of old audio, and if we have enough of it, attempt to
        // detect the wake word.
        self.recorded_audio.extend(buffer);

        let required_samples = WAKE_WORD_WINDOW_DURATION
            * self.recording_sample_rate
            * self.recording_channels;
        if self.recorded_audio.len() > required_samples {
            self.recorded_audio = self.recorded_audio[self.recorded_audio.len() - required_samples..].to_vec();
        }

        if self.recorded_audio.len() == required_samples {
            self.detect_wake_word()?;
        }

        Ok(())
    }

    fn detect_wake_word(&mut self) -> Result<(), Box<dyn Error>> {
        // We're going to try and detect the presence of the wake word in the recorded audio,
        // and if we see it, we'll transition the mode of the listener into "Start".
        let resampled_padded_audio = {
            let mut resampled_audio = self.resample_audio()?;
            if resampled_audio.len() < AUDIO_BUFFER_SIZE {
                resampled_audio.resize(AUDIO_BUFFER_SIZE, 0.0);
            }
            resampled_audio
        };
        let audio_slice = resampled_padded_audio
            .as_slice()
            .try_into()
            .unwrap();
        let audio_features = self.whisper_feature_extractor
            .extract_features(audio_slice);
        let likelihood = self.whisper_likelihood_model
            .estimate_likelihood(audio_features)?;

        if likelihood >= WAKE_WORD_ACTIVATION_THRESHOLD {
            self.mode = WakeWordListenerMode::Start;
            self.speaker_embedding = self.get_current_speaker_embeddings()?;
        }

        Ok(())
    }

    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let mut dot_product = 0.0;
        let mut a_norm = 0.0;
        let mut b_norm = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            dot_product += a_val * b_val;
            a_norm += a_val.powi(2);
            b_norm += b_val.powi(2);
        }
        1.0 - (dot_product / (a_norm.sqrt() * b_norm.sqrt()))
    }

    fn get_current_speaker_embeddings(&self) -> Result<Vec<f32>, Box<dyn Error>> {
        let required_samples = SPEAKER_EMBEDDING_WINDOW_DURATION * DESIRED_SAMPLE_RATE;
        let resampled_padded_audio = self.resample_audio()?;
        if resampled_padded_audio.len() < required_samples {
            Err("Not enough samples for speaker embedding.")?
        }

        let audio_slice = resampled_padded_audio[resampled_padded_audio.len() - required_samples..]
            .try_into()
            .unwrap();
        let audio_features = self.speech_brain_feature_extractor
            .extract_features(audio_slice);
        let embeddings = self.speech_brain_embedding_model
            .get_speaker_embeddings(&audio_features)?;
        let (_, embedding_dim) = embeddings.size2()?;
        Ok(
            (0..embedding_dim)
                .map(|i| embeddings.get(0).double_value(&[i]) as f32)
                .collect()
        )
    }

    fn handle_start_buffer(&mut self, buffer: Vec<f32>) -> Result<(), Box<dyn Error>> {
        // We're in "speech accumulation" mode, waiting for an endpoint.
        self.recorded_audio.extend(buffer);

        let new_speaker_embedding = self.get_current_speaker_embeddings()?;
        let distance = Self::cosine_distance(&self.speaker_embedding, &new_speaker_embedding);
        eprintln!("Distance: {}", distance);
        if distance >= SPEAKER_EMBEDDING_SIMILARITY_THRESHOLD {
            // We've detected an endpoint.
            self.mode = WakeWordListenerMode::Endpoint;
        }

        Ok(())
    }

    fn handle_endpoint_buffer(&mut self, buffer: Vec<f32>) -> Result<(), Box<dyn Error>> {
        // We're in "endpoint" mode, send request for a transcription and reset state.
        let recorded_audio = self.resample_audio()?;
        let transcription = self.whisper_transcriber.transcribe_audio(recorded_audio)?;
        println!("{}", transcription);

        self.recorded_audio = buffer;
        self.mode = WakeWordListenerMode::Background;

        Ok(())
    }
}
