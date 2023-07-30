use std::mem::{replace, take};
use std::ptr::null_mut;
use std::sync::{Mutex, mpsc};
use crate::wake_word_listener_state::{WakeWordListenerState, STREAM_FRAME_DURATION};

static mut AUDIO_BUFFER: Vec<f32> = Vec::new();
static AUDIO_BUFFER_UPDATING: Mutex<()> = Mutex::new(());
static AUDIO_BUFFER_PROCESSING: Mutex<()> = Mutex::new(());
pub static mut LISTENER_STATE: *mut WakeWordListenerState = null_mut();

pub fn update_audio_buffer(data: &[f32], sample_rate: usize, channels: usize, tx: mpsc::SyncSender<Vec<f32>>) {
    let lock = match AUDIO_BUFFER_UPDATING.try_lock() {
        Ok(_) => (),
        Err(_) => {
            eprintln!("Can't keep up! Audio buffer is already being updated.");
            return;
        },
    };

    let max_num_samples = ((sample_rate * channels) as f64 * (STREAM_FRAME_DURATION as f64 / 1000.0)) as usize;
    let audio_to_process = unsafe {
        let buffer = take(&mut AUDIO_BUFFER)
            .into_iter()
            .chain(data.iter().cloned())
            .collect::<Vec<_>>();
        if buffer.len() < max_num_samples {
            _ = replace(&mut AUDIO_BUFFER, buffer);
            None
        } else {
            let (to_process, to_keep) = buffer.split_at(max_num_samples);
            _ = replace(&mut AUDIO_BUFFER, to_keep.to_vec());
            Some(to_process.to_vec())
        }
    };

    drop(lock);

    if let Some(audio_to_process) = audio_to_process {
        match tx.try_send(audio_to_process) {
            Ok(_) => (),
            Err(err) => {
                eprintln!("Can't keep up! Audio processing buffer is not listening: {}", err);
            }
        }
    }
}

pub fn process_audio(buffer: Vec<f32>) {
    let lock = match AUDIO_BUFFER_PROCESSING.try_lock() {
        Ok(lock) => lock,
        Err(_) => {
            eprintln!("Can't keep up! Audio buffer is already processing.");
            return;
        }
    };

    let mut listener_state = unsafe {
        Box::from_raw(LISTENER_STATE)
    };

    match listener_state.process_audio(buffer) {
        Ok(_) => (),
        Err(err) => {
            eprintln!("Error processing audio: {}", err);
        }
    }

    eprintln!("Current mode: {:?}", listener_state.current_mode());

    unsafe {
        LISTENER_STATE = Box::into_raw(listener_state);
    }

    drop(lock);
}
