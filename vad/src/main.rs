extern crate core;

mod whisper_feature_extractor;
mod whisper_model;
mod speech_brain_embedding_model;
mod wake_word_listener;
mod speech_brain_feature_extractor;
mod whisper_transcription;
mod wake_word_listener_state;
mod microphone_stream;
mod stdin_stream;

use std::error::Error;
use tch::no_grad_guard;
use crate::microphone_stream::stream_microphone_audio;
use crate::stdin_stream::stream_stdin_audio;

fn _main() -> Result<(), Box<dyn Error>> {
    stream_microphone_audio();

    Ok(())
}

fn main() {
    let grad_guard = no_grad_guard();

    match _main() {
        Ok(_) => (),
        Err(e) => eprintln!("Error: {}", e),
    }

    drop(grad_guard);
}
