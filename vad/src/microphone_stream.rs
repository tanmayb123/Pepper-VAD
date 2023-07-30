use std::sync::mpsc;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crate::wake_word_listener_state::WakeWordListenerState;
use crate::wake_word_listener::{LISTENER_STATE, update_audio_buffer, process_audio};

pub fn stream_microphone_audio() {
    let host = cpal::default_host();

    let device = host.default_input_device()
        .expect("Failed to get default input device");
    let config = device.default_input_config()
        .expect("Failed to get default input config");

    let sample_rate = config.sample_rate().0 as usize;
    let channels = config.channels() as usize;
    let listener_state = Box::new(
        match WakeWordListenerState::new(sample_rate, channels) {
            Ok(state) => state,
            Err(err) => {
                eprintln!("Couldn't instantiate WakeWordListenerState: {}", err);
                return;
            }
        }
    );
    unsafe {
        LISTENER_STATE = Box::into_raw(listener_state);
    }

    let (tx, rx) = mpsc::sync_channel(1);
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _| {
            update_audio_buffer(data, sample_rate, channels, tx.clone());
        },
        move |err| {
            eprintln!("An error occurred on audio stream: {}", err);
        },
        None
    ).expect("Failed to build input stream");

    stream.play().expect("Failed to play stream");

    loop {
        let audio_to_process = match rx.recv() {
            Ok(audio) => audio,
            Err(err) => {
                eprintln!("Error receiving audio: {}", err);
                break;
            }
        };
        process_audio(audio_to_process);
    }
}

