use std::error::Error;
use std::fs::File;
use hound::WavWriter;
use reqwest::blocking::multipart::Form;
use reqwest::blocking::Client;
use serde::Deserialize;

pub struct WhisperTranscriber {
    api_key: String,
}

#[derive(Deserialize)]
struct WhisperResponse {
    text: String,
}

impl WhisperTranscriber {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
        }
    }

    pub fn transcribe_audio(&self, audio: Vec<f32>) -> Result<String, Box<dyn Error>> {
        let audio_file = File::create("audio.wav")?;

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = WavWriter::new(&audio_file, spec)?;
        for &sample in &audio {
            let amplitude = (sample * i16::max_value() as f32) as i16;
            writer.write_sample(amplitude)?;
        }
        writer.finalize()?;

        let form = Form::new()
            .text("model", "whisper-1")
            .file("file", "audio.wav")?;
        let client = Client::new();
        Ok(
            client.post("https://api.openai.com/v1/audio/transcriptions")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .multipart(form)
                .send()?
                .json::<WhisperResponse>()?
                .text
        )
    }
}
