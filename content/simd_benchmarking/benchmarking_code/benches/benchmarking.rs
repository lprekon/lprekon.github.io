#![feature(test)]
extern crate test;
use test::Bencher;

use rust_simd_becnhmarking::b_spline;

#[bench]
/// benchmark evaluating a degree-3 B-spline with 20 knots and 16 basis functions, over 100 different input values
fn bench_recursive_method(b: &mut Bencher) {
    let degree: usize = 3;
    let knots: Vec<f64> = (0..20).map(|x| x as f64).collect(); // 20 knots, ranging from 0 to 19
    let control_points: Vec<f64> = vec![1.0; 16]; // 16 control points, all 1.0
    let input_values: Vec<f64> = (0..100).map(|x| x as f64 / 10.0).collect(); // 100 input values, ranging from 0.0 to 9.9
    b.iter(|| {
        for x in input_values.iter() {
            let _ = b_spline(*x, &control_points, &knots, degree);
        }
    });
}
