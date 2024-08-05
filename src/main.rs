//use chrono::Utc;
// use rust_cuda::prelude::*;
use std::prelude::v1::*;
use std::time::Instant;
extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_61,code=sm_61")
        .file("cpp/mm.cu")
        .compile("mm.a");

    let n = 1024;

    let a: Vec<f32> = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];
    let mut c = vec![0.0f32; n * n];

    println!("Running on CPU...");
    let now = Instant::now();
    cpu_mm(&a, &b, &mut c, n);
    println!("CPU time: {:?}ms", now.elapsed().as_millis());

    // now = Instant::now();
    // let inp: ! = device.htod_copy(vec![1.0f32; 100])?;
    // let mut out = device.alloc_zeros::<f32>(100)?;
}

fn cpu_mm(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut value = 0.;
            for k in 0..n {
                value += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = value;
        }
    }
}

#[link(name = "matrixMulCUDA", kind = "static")]
extern "C" {
    fn matrixMulCUDA();
}
