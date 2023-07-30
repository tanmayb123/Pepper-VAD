use std::io::{self, Read};
use std::sync::mpsc;
use std::thread;
use crate::wake_word_listener_state::WakeWordListenerState;
use crate::wake_word_listener::{LISTENER_STATE, update_audio_buffer, process_audio};

pub fn stream_stdin_audio() {
    let sample_rate = 16000;
    let channels = 1;
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

    let handle = thread::spawn(move || {
        let mut buffer = [0; 4096];
        loop {
            match io::stdin().read_exact(&mut buffer) {
                Ok(_) => {
                    let audio_data: Vec<f32> = buffer
                        .chunks_exact(4)
                        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
                        .collect();
                    update_audio_buffer(&audio_data, sample_rate, channels, tx.clone());
                }
                Err(err) => {
                    eprintln!("Error reading audio data: {}", err);
                    break;
                }
            }
        }
    });

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

    handle.join().unwrap();
}
