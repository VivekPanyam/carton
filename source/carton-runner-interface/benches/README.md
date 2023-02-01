# Benchmark analysis

*Note: this analysis was done on 1/31/22. The results of benchmarks may have varied since then*

**TL;DR: An allocator that reuses previous allocations provides superior performance**

## Overview

This benchmark compares three things:

- Allocating storage for a tensor and filling with a non-default value.
  - (`inline_storage_without_pool` in the plots below)
- Allocating storage for a tensor using a pool of previously allocated and dropped items (falling back to a standard alloc if none are available) and filling with a non-default value.
  - (`inline_storage_with_pool` in the plots below)
- Filling an existing allocation with a non-default value.
  - (`fill_existing_tensor` in the plots below)

The purpose of the non-default fill is to ensure that the storage is not lazily allocated (with zero pages, for example). This also allows us to include data access times in the benchmark.

This is a useful benchmark under the following assumption:

- Repeated allocations of the same size/shape are common in inference workloads.

This is generally true for image-based models. Text models and other variable sized models are not currently included in this benchmark. They will be measured in the future.

## Results

As expected, reusing previous allocations leads to significant performance boosts, especially with larger tensors.

As you can see in the plot below, for large allocations (~ HD images), reusing previous allocations is ~14ms faster than allocating a new chunk of memory.

![](./lines.svg)

Here's a **log-log** plot of results of the same benchmark. For small allocations, the pool allocator is ~60 nanoseconds slower than the naive one and larger allocations approach the performance of "fill an existing allocation". This implies that the allocation time of the pool allocator approaches zero as the sizes increase.

![](./lines_log.svg)


So in the worst case, the pool allocator is 60 *nanoseconds* slower than an allocator without a pool and best case (in this benchmark) it's 14 *milliseconds* faster. Therefore, we enable the pool allocator by default.

## Future exploration

We should redo this analysis once a LRU cache is implemented within the pool. We should also test on allocation patterns that don't match the assumption above to ensure that performance is still good.
