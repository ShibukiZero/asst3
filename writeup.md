## Part 1: CUDA Warm-Up 1: SAXPY

### Q1

What performance do you observe compared to the sequential CPU-based
implementation of SAXPY (recall your results from saxpy on Program 5 from
Assignment 1)?

**Answer:**

All numbers below were measured on the same AWS `g5g.xlarge` instance, which
pairs an NVIDIA T4G GPU (40 SMs, ~15 GB, compute capability 7.5) with a 4-core
AWS Graviton2 (Neoverse-N1) CPU. To keep the comparison on a single machine, I
re-measured the Assignment 1 serial CPU saxpy here (single core, `N` = 20M,
3 warmup iterations + min of 10 runs) rather than reusing the original laptop
result. All bandwidths use the 3N convention (read `X`, read `Y`, write
`result` = 3 floats per element) so the CPU and GPU figures are directly
comparable; the GPU runs use the harness default of `N` = 100M, and saxpy
bandwidth is size-independent at these sizes because the kernel is memory-bound.

| Implementation | Hardware | Bandwidth (3N) |
|---|---|---:|
| Serial CPU saxpy (1 core) | Graviton2 | 23.8 GB/s |
| CUDA saxpy, kernel only | T4G | ~230 GB/s |
| CUDA saxpy, end-to-end (incl. PCIe transfer) | T4G | ~5.3 GB/s |

Two observations:

1. The GPU kernel alone is about **10x faster** than the serial CPU (230 vs
   23.8 GB/s), reflecting the T4G's much wider GDDR6 memory system. saxpy is
   purely memory-bound (2 flops per 12 bytes, arithmetic intensity
   ~0.17 flop/byte), so this gap is essentially the ratio of sustainable memory
   bandwidth between the two chips.

2. However, once the cost of copying `X` and `Y` to the GPU and the result back
   over PCIe is included, the effective throughput collapses to ~5.3 GB/s —
   about **4.5x slower** than a single CPU core, and the gap would widen further
   against a multi-threaded CPU version. Because saxpy does almost no arithmetic
   per byte, the one-time PCIe transfer dominates end-to-end time and the GPU's
   bandwidth advantage is entirely wasted. The CPU never pays this cost: its data
   already lives in the memory it computes from.

Conclusion: for an isolated, memory-bound saxpy, offloading to the GPU is a net
loss end-to-end. The GPU only pays off when the data already resides in device
memory (e.g., as one stage of a longer GPU pipeline, so the transfer is
amortized) or when the kernel performs enough arithmetic per byte to hide the
transfer cost.

---

### Q2

Compare and explain the difference between the results provided by two sets of
timers: timing only the kernel execution versus timing the entire process of
moving data to the GPU and back in addition to the kernel execution.

Are the bandwidth values observed roughly consistent with the reported
bandwidths available to the different components of the machine? Use the memory
bandwidth of an NVIDIA T4 GPU and the expected AWS memory bus bandwidth of
5.3 GB/s as references.

**Answer:**

The two timers measure very different things:

- **Kernel-only timer** (~4.86 ms, ~230 GB/s) wraps just the kernel launch plus
  `cudaDeviceSynchronize()`, so it measures GPU compute reading from and writing
  to device memory. It is bounded by the T4G's on-board memory bandwidth.
- **Whole-process timer** (~209 ms, ~5.3 GB/s) also includes the `cudaMemcpy` of
  `X` and `Y` host->device and `result` device->host. It is bounded by the
  host<->device PCIe link.

The ~43x gap between them is the ratio of on-board memory bandwidth to PCIe
bandwidth, and both numbers line up with the hardware:

- The NVIDIA T4 has ~320 GB/s of memory bandwidth. The measured kernel bandwidth
  of ~230 GB/s is about 72% of that, a reasonable sustained fraction for a simple
  streaming kernel (peak is rarely reached in practice).
- The effective bandwidth of ~5.3 GB/s matches the expected AWS host<->device bus
  bandwidth of 5.3 GB/s almost exactly. This is well below a 16-lane PCIe 3.0
  theoretical peak (~16 GB/s), because of chipset overheads and the use of
  pageable (non-pinned) host memory, which forces an extra staging copy.

So yes, both timers are consistent with the machine: the kernel timer reflects
device memory bandwidth, while the whole-process timer reflects the much slower
PCIe transfer, which dominates because the kernel itself is only ~5 ms of the
~209 ms total.

---

## Part 2: CUDA Warm-Up 2: Parallel Prefix-Sum

### Q1

Implement `exclusive_scan` in `scan/scan.cu` using the iterative upsweep /
downsweep parallel prefix-sum algorithm described in the handout.

**Answer:**

`exclusive_scan` runs on the host and drives the work-efficient scan by issuing
one kernel launch per level of the upsweep and downsweep phases. The input is
first copied into `result` and the tail is zero-padded up to the next power of
two, so the binary-tree algorithm is exact. The upsweep (`upsweep_kernel`)
reduces pairs up the tree; the root is then cleared and the downsweep
(`downsweep_kernel`) distributes the partial sums back down to produce the
exclusive prefix sums.

The key performance decision is that each level launches **exactly one thread
per active task, not one thread per element**. At distance `two_d` there are
`rounded_N / (2 * two_d)` tasks, and each kernel computes its task index as
`task = blockIdx.x * blockDim.x + threadIdx.x` and maps it directly to the array
slot `task * two_dplus1`. This keeps total work at O(N) and avoids the naive
pattern of launching N threads per level and masking off the inactive ones
(which would be especially wasteful at the top of the tree, where only a handful
of tasks remain). Successive kernel launches on the default stream are ordered,
so no explicit synchronization is needed between levels.

---

### Q2

Implement `find_repeats` in `scan/scan.cu`. Given an input array `A`, it should
return the list of all indices `i` for which `A[i] == A[i+1]`. The implementation
should use one or more calls to `exclusive_scan`.

**Answer:**

`find_repeats` uses the standard flag -> scan -> scatter pattern:

1. `make_flags_kernel` builds a 0/1 array of length `N-1` where `flag[i] = 1`
   iff `input[i] == input[i+1]`.
2. `exclusive_scan` over the flags yields `positions[i]` = the number of matches
   strictly before `i`, i.e. the output slot that match `i` should be written to.
3. `scatter_repeats_kernel` lets each thread whose flag is set write its index
   `i` into `output[positions[i]]`. Because the exclusive scan assigns each match
   a distinct, monotonically increasing slot, the writes never collide and the
   output stays in sorted index order.

The number of matches is recovered as `positions[N-2] + flags[N-2]` (the
exclusive-scan total). `exclusive_scan` does not modify `flags`, so the flag
array can be reused both for the scatter and for this final count.

---

### Q3

Report the correctness and performance results from:

```bash
./checker.py scan
./checker.py find_repeats
```

**Answer:**

Measured on an AWS `g5g.xlarge` instance (NVIDIA T4G GPU, CUDA 12.8). Times in ms.

| Test | Element Count | Ref Time | Student Time | Score |
|---|---:|---:|---:|---:|
| scan | 1000000 | 0.644 | 0.514 | 1.25 |
| scan | 10000000 | 8.943 | 8.319 | 1.25 |
| scan | 20000000 | 17.775 | 16.538 | 1.25 |
| scan | 40000000 | 35.242 | 33.026 | 1.25 |
| find_repeats | 1000000 | 1.028 | 0.814 | 1.25 |
| find_repeats | 10000000 | 11.978 | 10.343 | 1.25 |
| find_repeats | 20000000 | 21.458 | 19.330 | 1.25 |
| find_repeats | 40000000 | 41.597 | 37.428 | 1.25 |

Total scan score: 5.0 / 5.0
Total find_repeats score: 5.0 / 5.0

The student implementation is faster than the reference at every size, so all
tests earn full marks.

---

## Part 3: A Simple Circle Renderer

### Q1

Replicate the score table generated for your solution and specify which machine
you ran your code on.

Run the checker from the `render` directory:

```bash
./checker.py
```

**Answer:**

Machine:

| Scene Name | Ref Time (T_ref) | Your Time (T) | Score |
|---|---:|---:|---:|
| rgb |  |  |  |
| rand10k |  |  |  |
| rand100k |  |  |  |
| pattern |  |  |  |
| snowsingle |  |  |  |
| biglittle |  |  |  |
| rand1M |  |  |  |
| micro2M |  |  |  |

Total render score:

---

### Q2

Describe how you decomposed the rendering problem and how you assigned work to
CUDA thread blocks and threads, and maybe warps.

**Answer:**



---

### Q3

Describe where synchronization occurs in your solution.

**Answer:**



---

### Q4

What, if any, steps did you take to reduce communication requirements, such as
synchronization or main memory bandwidth requirements?

**Answer:**



---

### Q5

Briefly describe how you arrived at your final solution. What other approaches
did you try along the way, and what was wrong with them? Include what
measurements you performed to guide optimization.

**Answer:**



---

### Q6

Explain how your renderer preserves the two correctness requirements from the
handout:

- Atomicity: all image update operations must be atomic.
- Order: updates to the same pixel must be applied in circle input order.

**Answer:**



---

## Extra Credit

### Q1

If you implemented scan using an approach competitive with Thrust, describe it
and compare against the Thrust implementation.

**Answer:**



---

### Q2

If your renderer achieves significantly greater performance than required,
explain the approach thoroughly.

**Answer:**



---

### Q3

If you implemented a high-quality parallel CPU-only renderer, describe the
implementation, performance, and how it compares with your GPU solution.

**Answer:**



---

## Submission Checklist

- [ ] `writeup.pdf` generated from this writeup.
- [ ] Score tables copied into the writeup.
- [ ] Machine used for all measurements specified.
- [ ] Partner name and SUNet ID included, if working with a partner.
- [ ] `sh create_submission.sh` run from the assignment repository.
- [ ] Generated zip submitted to Gradescope.
