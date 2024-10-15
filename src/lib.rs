use std::f32::consts::PI;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum NeoResamplerError {
    #[error("Resampling error")]
    ResamplingError,
    #[error("Wrong frame rate; in: {0} Hz, out: {1} Hz")]
    WrongFrameRate(u32, u32),
    #[error("Wrong frame rate ratio; in: {0} Hz, out: {1} Hz")]
    WrongFrameRateRatio(u32, u32),
    #[error("Input buffer is too short: {0} samples")]
    WrongInputSize(usize),
    #[error("Output buffer is too short: {0} samples")]
    WrongOutputSize(usize),
    #[error("Given number of frames is too long: {0} frames")]
    WrongNumFrames(usize),
}

#[derive(Default)]
pub struct NeoResampler {
    input_frame_rate_hz: u32,
    output_frame_rate_hz: u32,
    num_input_frames_max: usize,
    num_output_frames_max: usize,
    num_channels: u16,
}

impl NeoResampler {
    pub fn new(
        input_frame_rate_hz: u32,
        output_frame_rate_hz: u32,
        num_frames_max: usize,
        num_channels: u16,
    ) -> Result<Self, NeoResamplerError> {
        let mut resampler = NeoResampler::default();
        resampler.prepare(
            input_frame_rate_hz,
            output_frame_rate_hz,
            num_frames_max,
            num_channels,
        )?;
        Ok(resampler)
    }

    pub fn prepare(
        &mut self,
        input_frame_rate_hz: u32,
        output_frame_rate_hz: u32,
        num_frames_max: usize,
        num_channels: u16,
    ) -> Result<(), NeoResamplerError> {
        Self::validate_args(input_frame_rate_hz, output_frame_rate_hz, num_channels)?;
        self.input_frame_rate_hz = input_frame_rate_hz;
        self.output_frame_rate_hz = output_frame_rate_hz;
        self.num_input_frames_max = num_frames_max;
        self.num_output_frames_max =
            Self::get_num_output_frames(num_frames_max, input_frame_rate_hz, output_frame_rate_hz);
        self.num_channels = num_channels;
        Ok(())
    }

    /// return frames actually written
    /// is only less if input_audio contains less samples then registered at prepare callback
    pub fn process(
        &self,
        input_audio: &[f32],
        num_frames: usize,
        output_audio: &mut [f32],
    ) -> Result<usize, NeoResamplerError> {
        if num_frames > self.num_input_frames_max {
            return Err(NeoResamplerError::WrongNumFrames(num_frames));
        }
        if input_audio.len() != num_frames * self.num_channels as usize {
            return Err(NeoResamplerError::WrongInputSize(input_audio.len()));
        }

        let num_output_frames = Self::get_num_output_frames(
            num_frames,
            self.input_frame_rate_hz,
            self.output_frame_rate_hz,
        );
        if output_audio.len() < num_output_frames * self.num_channels as usize {
            return Err(NeoResamplerError::WrongOutputSize(output_audio.len()));
        }

        for input_frame_idx in 0..num_output_frames {
            for channel_idx in 0..self.num_channels {
                let output_frame_idx = (input_frame_idx as f32 * self.input_frame_rate_hz as f32)
                    / self.output_frame_rate_hz as f32;
                *Self::get_mut(
                    output_audio,
                    input_frame_idx,
                    channel_idx as usize,
                    self.num_channels as usize,
                ) = Self::compute_sample(
                    input_audio,
                    output_frame_idx,
                    channel_idx as usize,
                    self.num_channels as usize,
                );
            }
        }
        Ok(num_output_frames)
    }

    pub fn num_output_frames_max(&self) -> usize {
        self.num_output_frames_max
    }

    const KERNEL_A: i32 = 5;

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_sign_loss)]
    fn get_num_output_frames(
        num_input_frames: usize,
        input_frame_rate_hz: u32,
        output_frame_rate_hz: u32,
    ) -> usize {
        ((num_input_frames as f64 * f64::from(output_frame_rate_hz)
            / f64::from(input_frame_rate_hz))
        .ceil()) as usize
    }

    fn validate_args(
        input_frame_rate_hz: u32,
        output_frame_rate_hz: u32,
        num_channels: u16,
    ) -> Result<(), NeoResamplerError> {
        if input_frame_rate_hz == 0 || output_frame_rate_hz == 0 {
            return Err(NeoResamplerError::WrongFrameRate(
                input_frame_rate_hz,
                output_frame_rate_hz,
            ));
        }
        if num_channels == 0 {
            return Err(NeoResamplerError::ResamplingError);
        }
        if (input_frame_rate_hz > output_frame_rate_hz)
            && (f64::from(input_frame_rate_hz) / f64::from(output_frame_rate_hz) > 256.0)
        {
            return Err(NeoResamplerError::WrongFrameRateRatio(
                input_frame_rate_hz,
                output_frame_rate_hz,
            ));
        }
        if f64::from(output_frame_rate_hz) / f64::from(input_frame_rate_hz) > 256.0 {
            return Err(NeoResamplerError::WrongFrameRateRatio(
                input_frame_rate_hz,
                output_frame_rate_hz,
            ));
        }
        Ok(())
    }

    fn lanczos_kernel(x: f32, a: f32) -> f32 {
        if float_cmp::approx_eq!(f32, x, 0.0_f32) {
            return 1.0;
        }
        if -a <= x && x < a {
            return (a * (PI * x).sin() * (PI * x / a).sin()) / (PI * PI * x * x);
        }
        0.0
    }

    fn get<T: Copy>(v: &[T], frame: usize, channel_idx: usize, num_channels: usize) -> T {
        v[frame * num_channels + channel_idx]
    }

    fn get_mut<T>(v: &mut [T], frame: usize, channel_idx: usize, num_channels: usize) -> &mut T {
        &mut v[frame * num_channels + channel_idx]
    }

    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn compute_sample(
        input_audio: &[f32],
        frame_idx: f32,
        channel_idx: usize,
        num_channels: usize,
    ) -> f32 {
        let num_input_frames: usize = input_audio.len() / num_channels as usize;
        let a: f32 = Self::KERNEL_A as f32;
        let x_floor = frame_idx as i64;
        let i_start = x_floor - a as i64 + 1;
        let i_end = x_floor + a as i64 + 1;
        let mut the_sample: f32 = 0.0_f32;
        for i in i_start..i_end {
            if (i as usize) < num_input_frames {
                the_sample += Self::get(input_audio, i as usize, channel_idx, num_channels)
                    * Self::lanczos_kernel(frame_idx - i as f32, a);
            }
        }
        the_sample
    }
}

#[cfg(test)]
mod tests {
    use babycat::{constants::RESAMPLE_MODE_BABYCAT_LANCZOS, Signal, Waveform, WaveformArgs};
    use rand::Rng;

    use super::*;

    #[test]
    fn test_small() {
        let num_frames = 64;
        let input_sr = 48000;
        let output_sr = 16000;

        let mut rng = rand::thread_rng();
        let input_audio: Vec<f32> = (0..num_frames)
            .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
            .collect();

        let waveform = Waveform::from_interleaved_samples(input_sr, 1, &input_audio);
        let expected_output = waveform
            .resample_by_mode(output_sr, RESAMPLE_MODE_BABYCAT_LANCZOS)
            .unwrap();

        let mut resampler = NeoResampler::default();
        resampler
            .prepare(input_sr, output_sr, num_frames, 1)
            .unwrap();

        let mut generated_output = vec![0.0; resampler.num_output_frames_max()];
        let frames_written = resampler
            .process(&input_audio, num_frames, &mut generated_output)
            .unwrap();
        assert_eq!(frames_written, resampler.num_output_frames_max());

        for (exp, gen) in expected_output
            .to_interleaved_samples()
            .iter()
            .zip(generated_output.iter())
        {
            assert!(float_cmp::approx_eq!(f32, *exp, *gen));
        }
    }

    #[test]
    fn test_resampler() {
        let waveform_args = WaveformArgs {
            ..Default::default()
        };
        let waveform = match Waveform::from_file("flight_orig.wav", waveform_args) {
            Ok(w) => w,
            Err(err) => {
                println!("Decoding error: {}", err);
                return;
            }
        };

        let samples = waveform.to_interleaved_samples();

        let num_frames = 512;

        let mut resampler = NeoResampler::default();

        let input_frame_rate_hz = waveform.frame_rate_hz();
        let output_frame_rate_hz = 16000;
        let num_channels = waveform.num_channels();
        resampler
            .prepare(
                input_frame_rate_hz,
                output_frame_rate_hz,
                num_frames,
                num_channels,
            )
            .unwrap();

        for input_audio in samples.chunks_exact(num_frames) {
            let mut generated_output = vec![0.0; resampler.num_output_frames_max()];
            let frames_written = resampler
                .process(input_audio, num_frames, &mut generated_output)
                .unwrap();
            assert_eq!(frames_written, resampler.num_output_frames_max());

            let waveform = Waveform::from_interleaved_samples(input_frame_rate_hz, 1, &input_audio);
            let expected_output = waveform
                .resample_by_mode(output_frame_rate_hz, RESAMPLE_MODE_BABYCAT_LANCZOS)
                .unwrap();

            for (exp, gen) in expected_output
                .to_interleaved_samples()
                .iter()
                .zip(generated_output.iter())
            {
                assert!(float_cmp::approx_eq!(f32, *exp, *gen));
            }
        }
    }
}
