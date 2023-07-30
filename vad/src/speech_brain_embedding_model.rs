use tch::{Kind, TchError, Tensor};
use crate::speech_brain_feature_extractor::N_MELS;

pub struct SpeechBrainEmbeddingModel {
    model: tch::CModule,
}

impl SpeechBrainEmbeddingModel {
    pub fn new() -> Result<SpeechBrainEmbeddingModel, TchError> {
        let model_bytes = include_bytes!("speaker-embeddings.pt");
        let mut model_reader = std::io::Cursor::new(model_bytes);
        let model = tch::CModule::load_data(&mut model_reader)?;
        Ok(
            SpeechBrainEmbeddingModel {
                model,
            }
        )
    }

    pub fn get_speaker_embeddings(&self, audio: &[f32]) -> Result<Tensor, TchError> {
        let frames = audio.len() / N_MELS;
        let audio_tensor = Tensor::from_slice(audio)
            .to_kind(Kind::Float)
            .reshape(&[1, frames as i64, N_MELS as i64]);
        let output = self.model.forward_ts(&[audio_tensor])?;
        Ok(output.get(0))
    }
}