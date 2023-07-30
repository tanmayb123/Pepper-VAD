use std::ptr::copy_nonoverlapping;
use tch::{Device, Kind, TchError, Tensor};
use crate::whisper_feature_extractor::{FEATURES_SIZE, N_FRAMES, N_MELS};

pub struct WhisperLikelihoodModel {
    model: tch::CModule,
    features_tensor: Tensor,
}

impl WhisperLikelihoodModel {
    pub fn new() -> Result<WhisperLikelihoodModel, TchError> {
        let model_bytes = include_bytes!("whisper-likelihood.pt");
        let mut model_reader = std::io::Cursor::new(model_bytes);
        let model = tch::CModule::load_data(&mut model_reader)?;
        let features_tensor = Tensor::zeros(
            &[1, N_MELS as i64, N_FRAMES as i64],
            (Kind::Float, Device::Cpu)
        );
        Ok(
            WhisperLikelihoodModel {
                model,
                features_tensor,
            }
        )
    }

    pub fn estimate_likelihood(&self, features: &[f32; FEATURES_SIZE]) -> Result<f32, TchError> {
        unsafe {
            copy_nonoverlapping(
                features.as_ptr(),
                self.features_tensor.data_ptr() as *mut f32,
                FEATURES_SIZE,
            );
        }
        let output = self.model.forward_ts(&[&self.features_tensor])?;
        Ok(output.double_value(&[]) as f32)
    }
}