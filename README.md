# NeoResampler

A resampler written for realtime-audio processors.

Based on the resampler in [babycat](https://github.com/babycat-io/babycat)  which is implemented based on this reference https://en.wikipedia.org/wiki/Lanczos_resampling.

## Example

- Initialize as Default
```rust
let mut resampler = NeoResampler::default();
let mut resampler_output = vec![0_f32; 0];
```

- Prepare for audio processing
```rust
// set your audio settings
resampler.prepare(
    input_frame_rate_hz,
    output_frame_rate_hz,
    num_frames,
    num_channels,
).unwrap();

// resize the output buffer to the expected maximum number of output frames
resampler_output.resize(resampler.num_output_frames_max(), 0.0);
```

- Call in your audio process
```rust
// The buffers are expected to be interleaved if they contain multiple channels.
// In plug-ins DAWs can give you sometimes less input samples, than agreed on.
// Only in this case, the num_samples_written will be smaller than the output buffer is long.
// If you know you always gonna call it with the agreed number of frames from the prepare function,
// you can ignore the returned num_samples_written.
let num_samples_written = resampler.process(&input_audio, num_frames, &mut resampler_output).unwrap();
```
