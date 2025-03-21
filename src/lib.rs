use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use pyo3::prelude::*;
use rustpotter::{Rustpotter, RustpotterConfig, RustpotterDetection};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender, channel};

/// Преобразует аудио данные:
/// 1. Если входной сигнал многоканальный, а требуется моно, то усредняет каналы.
/// 2. Выполняет ресемплирование, если входная частота дискретизации отличается от требуемой.
fn convert_audio(
    data: &[f32],
    input_channels: u16,
    desired_channels: u16,
    input_sample_rate: u32,
    desired_sample_rate: u32,
) -> Vec<f32> {
    // Если входных каналов больше одного и требуется моно, усредняем сэмплы по кадру.
    let mono_data: Vec<f32> = if input_channels > 1 && desired_channels == 1 {
        data.chunks(input_channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / input_channels as f32)
            .collect()
    } else {
        data.to_vec()
    };

    // Если входная и требуемая частоты дискретизации отличаются, выполняем ресемплирование.
    if input_sample_rate != desired_sample_rate {
        let ratio = desired_sample_rate as f32 / input_sample_rate as f32;
        let original_frames = mono_data.len();
        let new_length = (original_frames as f32 * ratio).round() as usize;
        let mut resampled = Vec::with_capacity(new_length);
        for i in 0..new_length {
            let orig_pos = i as f32 / ratio;
            let index = orig_pos.floor() as usize;
            let frac = orig_pos - (index as f32);
            if index + 1 < original_frames {
                let sample = mono_data[index] * (1.0 - frac) + mono_data[index + 1] * frac;
                resampled.push(sample);
            } else {
                resampled.push(mono_data[index]);
            }
        }
        resampled
    } else {
        mono_data
    }
}

fn convert_format(fmt: cpal::SampleFormat) -> rustpotter::SampleFormat {
    match fmt {
        cpal::SampleFormat::F32 => rustpotter::SampleFormat::F32,
        cpal::SampleFormat::I16 => rustpotter::SampleFormat::I16,
        cpal::SampleFormat::U16 => rustpotter::SampleFormat::F32,
        _ => rustpotter::SampleFormat::F32,
    }
}

#[pyclass(unsendable)]
struct AudioStream {
    buffer_size: usize,
    desired_channels: u16,
    desired_sample_rate: u32,
    stream: Option<cpal::Stream>,
    receiver: Option<Receiver<Vec<f32>>>,
    rustpotter_obr: Option<Rustpotter>,
    sample_format: Option<cpal::SampleFormat>,
    audio_buffer: VecDeque<f32>,
}

#[pymethods]
impl AudioStream {
    #[new]
    fn new(buffer_size: usize, desired_channels: u16, desired_sample_rate: u32) -> Self {
        AudioStream {
            buffer_size,
            desired_channels,
            desired_sample_rate,
            stream: None,
            receiver: None,
            rustpotter_obr: None,
            sample_format: None,
            audio_buffer: VecDeque::new(),
        }
    }

    fn start(&mut self) -> PyResult<()> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Нет доступного устройства ввода")
        })?;
        let config_result = device.default_input_config().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Ошибка получения конфигурации ввода: {}",
                e
            ))
        })?;

        let sample_format = config_result.sample_format();
        self.sample_format = Some(sample_format);

        let config: cpal::StreamConfig = config_result.into();

        let input_channels = config.channels;
        let input_sample_rate = config.sample_rate.0;
        let desired_channels = self.desired_channels;
        let desired_sample_rate = self.desired_sample_rate;

        let (sender, receiver): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = channel();

        let err_fn = |err| eprintln!("Ошибка в аудио потоке: {}", err);

        let stream = match sample_format {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    let processed = convert_audio(
                        data,
                        input_channels,
                        desired_channels,
                        input_sample_rate,
                        desired_sample_rate,
                    );
                    let _ = sender.send(processed);
                },
                err_fn,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    let float_data: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    let processed = convert_audio(
                        &float_data,
                        input_channels,
                        desired_channels,
                        input_sample_rate,
                        desired_sample_rate,
                    );
                    let _ = sender.send(processed);
                },
                err_fn,
                None,
            ),
            cpal::SampleFormat::U16 => device.build_input_stream(
                &config,
                move |data: &[u16], _| {
                    let float_data: Vec<f32> = data
                        .iter()
                        .map(|&s| (s as f32 - 32768.0) / 32768.0)
                        .collect();
                    let processed = convert_audio(
                        &float_data,
                        input_channels,
                        desired_channels,
                        input_sample_rate,
                        desired_sample_rate,
                    );
                    let _ = sender.send(processed);
                },
                err_fn,
                None,
            ),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Unsupported sample format",
                ));
            }
        }
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Не удалось создать аудио поток: {}",
                e
            ))
        })?;

        stream.play().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Не удалось запустить аудио поток: {}",
                e
            ))
        })?;

        self.stream = Some(stream);
        self.receiver = Some(receiver);

        Ok(())
    }

    fn clean_chunk(&self) -> () {
        ()
    }

    fn get_audio_chunk(&self) -> PyResult<Vec<f32>> {
        if let Some(ref receiver) = self.receiver {
            match receiver.try_recv() {
                Ok(chunk) => Ok(chunk),
                Err(_) => Ok(vec![]),
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Аудиопоток не запущен",
            ))
        }
    }

    fn load_model(&mut self, model_path: &str) {
        if let Some(fmt) = self.sample_format {
            println!("Выбранный sample_format: {:?}", fmt);
        } else {
            println!("sample_format не установлен");
        }
        let mut config = RustpotterConfig::default();
        config.fmt.sample_rate = self.desired_sample_rate as usize;
        let format = convert_format(self.sample_format.unwrap());
        config.fmt.sample_format = format;
        config.fmt.channels = 1;
        self.rustpotter_obr = Some(Rustpotter::new(&config).unwrap());
        let key = Path::new(model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        self.rustpotter_obr
            .as_mut()
            .unwrap()
            .add_wakeword_from_file(key, model_path)
            .unwrap();
        println!(
            "rustpotter_buffer_size {}",
            self.rustpotter_obr
                .as_ref()
                .unwrap()
                .get_samples_per_frame()
        );
    }

    fn detect(&mut self, chunk: Vec<f32>) -> PyResult<Vec<Option<Detection>>> {
        let rp: &mut Rustpotter = self
            .rustpotter_obr
            .as_mut()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not loaded"))?;

        self.audio_buffer.extend(chunk);

        let required_samples = rp.get_samples_per_frame();
        let mut detections = Vec::new();

        while self.audio_buffer.len() >= required_samples {
            let mut frame: Vec<f32> = Vec::with_capacity(required_samples);
            for _ in 0..required_samples {
                frame.push(self.audio_buffer.pop_front().unwrap());
            }
            let detection_opt = rp.process_samples(frame);
            let det_py = detection_opt.map(|det| Detection {
                name: det.name,
                score: det.score,
            });
            detections.push(det_py);
        }

        Ok(detections)
    }
}

#[pyclass]
struct Detection {
    name: String,
    score: f32,
}

#[pymethods]
impl Detection {
    #[new]
    fn new(name: String, score: f32) -> Self {
        Detection { name, score }
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    fn score(&self) -> f32 {
        self.score
    }
}

#[pymodule]
fn wakeword(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AudioStream>()?;
    m.add_class::<Detection>()?;
    Ok(())
}
