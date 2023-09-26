// Copyright 2023 Vivek Panyam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! This benchmark measures tensor allocation overhead
use carton_macros::for_each_numeric_carton_type;
use carton_runner_interface::_only_public_for_benchmarks_do_not_use::{
    alloc_tensor_inline, alloc_tensor_no_pool_inline, alloc_tensor_no_pool_shm, alloc_tensor_shm,
    InlineAllocator, InlineTensorStorage, SHMAllocator, SHMTensorStorage, TypedAlloc,
};
use criterion::{
    criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};

/// Note: this benchmark allocates *and fills* a tensor. Otherwise for large allocs, the allocation
/// ends up being very fast because it just allocates zero pages (and then the first writes are slow)
fn typed_alloc_benchmark<T: Clone + Default, U: Measurement>(
    name: &str,
    group: &mut BenchmarkGroup<U>,
    shape: &Vec<u64>,
    fill_value: T,
) where
    InlineAllocator: TypedAlloc<T, Output = InlineTensorStorage>,
    SHMAllocator: TypedAlloc<T, Output = SHMTensorStorage>,
{
    let numel = shape.iter().product::<u64>();
    let size_bytes = std::mem::size_of::<T>() as u64 * numel;
    group.throughput(Throughput::Bytes(size_bytes));
    group.bench_with_input(
        BenchmarkId::new(
            "inline_storage_with_pool",
            name.to_owned() + "_" + std::any::type_name::<T>(),
        ),
        shape,
        |b, shape| {
            b.iter(|| {
                let mut t = alloc_tensor_inline::<T>(shape.clone());
                t.view_mut()
                    .as_slice_mut()
                    .unwrap()
                    .fill(fill_value.clone());
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "inline_storage_without_pool",
            name.to_owned() + "_" + std::any::type_name::<T>(),
        ),
        shape,
        |b, shape| {
            b.iter(|| {
                let mut t = alloc_tensor_no_pool_inline::<T>(shape.clone());
                t.view_mut()
                    .as_slice_mut()
                    .unwrap()
                    .fill(fill_value.clone());
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "fill_existing_tensor",
            name.to_owned() + "_" + std::any::type_name::<T>(),
        ),
        shape,
        |b, shape| {
            let mut t = alloc_tensor_inline::<T>(shape.clone());
            b.iter(|| {
                t.view_mut()
                    .as_slice_mut()
                    .unwrap()
                    .fill(fill_value.clone());
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "shm_storage_with_pool",
            name.to_owned() + "_" + std::any::type_name::<T>(),
        ),
        shape,
        |b, shape| {
            b.iter(|| {
                let mut t = alloc_tensor_shm::<T>(shape.clone());
                t.view_mut()
                    .as_slice_mut()
                    .unwrap()
                    .fill(fill_value.clone());
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "shm_storage_without_pool",
            name.to_owned() + "_" + std::any::type_name::<T>(),
        ),
        shape,
        |b, shape| {
            b.iter(|| {
                let mut t = alloc_tensor_no_pool_shm::<T>(shape.clone());
                t.view_mut()
                    .as_slice_mut()
                    .unwrap()
                    .fill(fill_value.clone());
            })
        },
    );
}

fn alloc_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Alloc");
    // let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    // group.plot_config(plot_config);

    for (name, shape) in [
        // These sizes are selected to roughly provide exponential spacing across samples
        // (spaced roughly ~16x apart because each item is tested with types between 1 byte and 8 bytes)
        ("image", vec![1, 1200, 1920, 3]),
        ("small_image", vec![1, 480, 320, 3]),
        ("square", vec![160, 160]),
        ("small", vec![5, 320]),
        ("one_d", vec![100]),
        ("tiny", vec![8]),
    ]
    .iter()
    {
        // NOTE: this only tests numeric types
        for_each_numeric_carton_type! {
            $(
                typed_alloc_benchmark::<$RustType, _>(name, &mut group, shape, 1 as _);
            )*
        }
    }
    group.finish();
}

criterion_group!(benches, alloc_benchmark);
criterion_main!(benches);
