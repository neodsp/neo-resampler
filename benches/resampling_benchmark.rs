use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neo_resampler::NeoResampler;
use rand::Rng;

pub fn resampling_benchmark(c: &mut Criterion) {
    let resampler = NeoResampler::new(44100, 16000, 512, 1).unwrap();
    let mut rng = rand::thread_rng();
    let in_buffer: Vec<f32> = (0..512).map(|_| rng.gen::<f32>()).collect();
    let mut out_buffer = vec![0.0; resampler.num_output_frames_max()];
    c.bench_function("resampling_benchmark 44.1 -> 16, 512", |b| {
        b.iter(|| resampler.process(&in_buffer, 512, black_box(&mut out_buffer)))
    });
}

criterion_group!(benches, resampling_benchmark);
criterion_main!(benches);
