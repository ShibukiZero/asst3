## Part 1: CUDA Warm-Up 1: SAXPY

### Q1

What performance do you observe compared to the sequential CPU-based
implementation of SAXPY (recall your results from saxpy on Program 5 from
Assignment 1)?

**Answer:**

All numbers below were measured on the same AWS `g5g.xlarge` instance, which
pairs an NVIDIA T4G GPU (40 SMs, ~15 GB, compute capability 7.5) with a 4-core
AWS Graviton2 (Neoverse-N1) CPU. To keep the comparison on a single machine, the
Assignment 1 serial CPU saxpy was re-measured here (single core, `N` = 20M,
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

`exclusive_scan` uses a **block-local shared-memory scan plus a recursive scan of
the per-block totals**, rather than the naive one-kernel-per-tree-level approach.
A naive version launches ~`2*log2(N)` kernels and streams the whole array through
global memory once per level; for `N` = 40M that is ~50 global-memory passes and
dominates runtime. The block-scan version touches global memory only a couple of
times, which made it ~11.5x faster than the provided reference at 40M (see the
profiling-backed comparison and the further tuning in Extra Credit Q1).

The host driver `scan_device(in, out, n)` runs three phases:

1. **Block scan** (`block_scan_kernel`): the array is split into chunks of
   `ELEMENTS_PER_BLOCK` = 512 elements (256 threads, 2 elements each). Each block
   loads its chunk of `in` into shared memory, runs the work-efficient Blelloch
   upsweep/downsweep entirely in shared memory (with `CONFLICT_FREE_OFFSET`
   padding to avoid bank conflicts), and writes the scanned chunk to `out` plus
   its chunk total into `block_sums`. Reading `in` and writing `out` directly
   avoids an extra device-to-device copy of the whole array.
2. **Scan the block totals**: `scan_device` recurses on `block_sums` so that
   `block_sums[i]` becomes the exclusive offset that chunk `i` needs. For 40M
   elements the recursion is only ~3 levels deep (40M -> ~78k -> ~153 -> 1).
3. **Add offsets** (`add_block_offsets_kernel`): each block adds its scanned
   offset to every element of its chunk, yielding the global exclusive scan.

Boundary handling is done with bounds checks (tail elements past `n` load 0 and
are not written back), so no power-of-two padding of the input is required.
Kernel launches on the default stream are ordered, so no explicit host-side
synchronization is needed between phases.

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
| scan | 1000000 | 0.650 | 0.426 | 1.25 |
| scan | 10000000 | 8.957 | 1.014 | 1.25 |
| scan | 20000000 | 17.690 | 1.693 | 1.25 |
| scan | 40000000 | 35.275 | 3.068 | 1.25 |
| find_repeats | 1000000 | 1.053 | 0.722 | 1.25 |
| find_repeats | 10000000 | 12.003 | 2.915 | 1.25 |
| find_repeats | 20000000 | 21.444 | 4.385 | 1.25 |
| find_repeats | 40000000 | 41.602 | 8.329 | 1.25 |

Total scan score: 5.0 / 5.0
Total find_repeats score: 5.0 / 5.0

The block-scan implementation is many times faster than the reference at every
size (e.g. ~11.5x at 40M for scan), so all tests earn full marks.

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

Machine: AWS `g5g.xlarge` (NVIDIA T4G GPU, compute capability 7.5, 40 SMs,
~15 GB, CUDA 12.8). Times in ms.

| Scene Name | Ref Time (T_ref) | Your Time (T) | Score |
|---|---:|---:|---:|
| rgb | 0.2624 | 0.2627 | 9 |
| rand10k | 3.0574 | 1.9167 | 9 |
| rand100k | 29.7249 | 17.7517 | 9 |
| pattern | 0.4097 | 0.3006 | 9 |
| snowsingle | 19.7341 | 6.5476 | 9 |
| biglittle | 15.3124 | 15.0094 | 9 |
| rand1M | 241.9320 | 84.4519 | 9 |
| micro2M | 471.6679 | 145.0002 | 9 |

Total render score: 72 / 72

The solution is faster than the reference on every scene except the two that are
already at a floor (rgb is bound by the unavoidable image read/write; biglittle
is throughput-bound on a few huge circles). On the circle-heavy scenes it is
significantly faster than the reference (e.g. snowsingle ~3.0x, micro2M ~3.3x,
rand1M ~2.9x).

---

### Q2

Describe how you decomposed the rendering problem and how you assigned work to
CUDA thread blocks and threads, and maybe warps.

**Answer:**

The decomposition is **by pixels (tiles), not by circles**. The starter kernel
parallelized over circles (one thread per circle), which is what forces the
non-atomic, out-of-order image updates. Instead, the screen is partitioned into
`TILE_DIM x TILE_DIM` = 32x32 pixel tiles, and:

- **one thread block per tile**, and
- **one thread per pixel** within the tile (1024 threads = 32x32).

Each thread block has a **dual role** over the same 1024 threads:

- During *culling*, the threads act as **circle testers**: the block walks the
  circle array in batches of `SCAN_BLOCK_DIM` = 1024, and in each batch thread
  `i` tests circle `(batchStart + i)` against the tile's bounding box with
  `circleInBox`, producing a 0/1 hit flag.
- During *shading*, the same threads act as **pixels**: thread `i` owns one pixel
  and blends circles into a private `float4` register accumulator.

For each batch, the hit flags are compacted into an ordered, dense list of
circle indices using the shared-memory exclusive scan (`exclusiveScan.cu_inl`):
the scan turns the flag array into per-thread output slots, and each hitting
thread scatters its circle index into the shared `hitList`. Every pixel then
loops over just that batch's hit list (the circles that actually touch the tile),
in index order, instead of over all N circles. This is what cuts the per-pixel
work from O(N) to O(circles touching the tile).

The natural unit is the block/tile; warps matter only in that the scan requires
a warp-sorted linear thread index (`threadIdx.y * blockDim.x + threadIdx.x`), so
that 32 consecutive linear indices form a warp.

---

### Q3

Describe where synchronization occurs in your solution.

**Answer:**

All synchronization is **intra-block** (`__syncthreads()`); there is no
cross-block synchronization and no atomics. Each circle batch has four barriers,
because the cull / scan / scatter / shade steps share the same shared-memory
buffers and each step depends on all threads finishing the previous one:

1. After every thread writes its hit flag, before the scan reads the flag array.
2. After the scan, before reading the offsets — the provided `sharedMemExclusiveScan`
   has internal barriers but **no trailing barrier** after it writes its output,
   so one is needed before any thread reads another thread's offset (and the
   batch hit count at the last index).
3. After the scatter into `hitList`, before the pixels read it for shading.
4. After shading, before the next batch overwrites the shared buffers.

These barriers are correct because the loop bound (`numCircles`) is uniform
across the block, so every thread runs the same number of batches and reaches
every barrier together. The number of barriers (4 per batch) is also why tile
size matters for performance — see Q4/Q5.

---

### Q4

What, if any, steps did you take to reduce communication requirements, such as
synchronization or main memory bandwidth requirements?

**Answer:**

- **Register accumulator, one image write per pixel.** The starter `shadePixel`
  did a global read-modify-write of the image for every (circle, pixel)
  contribution. Each thread instead accumulates its pixel into a `float4` held in
  registers and writes it to global memory exactly once at the end. This removes
  almost all of the image-memory traffic and is also what makes the update atomic
  (Q6).

- **Culling removes redundant work, which was the real bottleneck.** Profiling
  the naive pixel-parallel version with Nsight Compute showed it was *not*
  DRAM-bound (DRAM throughput < 0.15% on every circle-heavy scene) but
  compute/L1-bound: the circle data stays in cache, so the cost was the sheer
  number of (pixel, circle) iterations. Building a per-tile hit list cuts those
  iterations from O(pixels x N) to roughly O(pixels x circles-per-tile), directly
  attacking that bottleneck. The hit list lives in shared memory, so the
  shade-phase reads of circle indices never go to global memory.

- **Tile size chosen to minimize synchronization overhead.** Each batch costs a
  shared-memory scan plus four `__syncthreads()`, and the number of batches is
  `N / SCAN_BLOCK_DIM`. Using the largest legal tile (32x32 -> batch of 1024)
  minimizes the batch count and therefore the total scan + barrier overhead;
  profiling the 16x16 version showed it was partly stalled (no pipe saturated) on
  exactly this overhead.

Circle position/radius/color was deliberately **not** cached in shared memory
for the shade phase. It was an option (it would relieve L1, which is co-saturated),
but it costs shared memory and occupancy, and the profiling showed culling alone
already reached and exceeded the reference, so the extra trade was unnecessary.

---

### Q5

Briefly describe how you arrived at your final solution. What other approaches
did you try along the way, and what was wrong with them? Include what
measurements you performed to guide optimization.

**Answer:**

The solution was reached in three measured steps:

1. **Correct, naive pixel-parallel baseline.** First the axis was flipped from
   circles to pixels: one thread per pixel, each looping over all circles in
   order and accumulating in a register. This is trivially correct (Q6) and
   scored 26/72 — full marks on tiny scenes (rgb) but failing the performance
   bar on circle-heavy scenes because its work is O(pixels x circles).

2. **Profile to find the real bottleneck.** Nsight Systems confirmed the
   render kernel dominated runtime; Nsight Compute (SpeedOfLight) on all
   eight scenes showed the bottleneck was compute + L1, *not* DRAM (DRAM
   throughput was ~0.02-0.13%). This corrected the initial assumption that it
   would be memory-bound from re-reading circle data: the circle arrays stay in
   cache (all warps march through the array together), so the cost is the number
   of iterations, not memory bandwidth. The fix therefore had to *reduce
   iterations*, which is exactly what tiling + culling does — not caching.

3. **Tiled culling, then a tile-size sweep.** The per-tile cull / scan /
   scatter / shade kernel was added, which reached 72/72. Re-profiling the 16x16 version
   showed it was no longer saturating any pipe on sparse scenes (compute ~67-71%,
   L1 ~74-76%), with the stalls pointing at per-batch overhead (the scan + 4
   barriers, run `N/256` times; micro2M has ~7800 batches). Hypothesis: a larger
   tile means a larger batch, fewer batches, and less fixed overhead. A sweep
   over 8/16/32 confirmed it monotonically — 8x8 was much slower (4-5/9 on many
   scenes), 32x32 much faster — so the tile was set to 32x32 (the largest legal
   size, since the scan caps at 1024 threads). This gave ~3x speedups on the
   circle-heavy scenes (micro2M 502 -> 145 ms) and put the renderer well past the
   reference.

Approaches considered and rejected: a per-circle approach with atomic/locked
image updates (fails the ordering requirement and contends badly); a dense
length-N flag array per tile instead of a compacted list (avoids write conflicts
but keeps the per-pixel loop at O(N) and does not fit in shared memory at scale);
and shared-memory caching of circle data (a real but unnecessary trade, see Q4).

---

### Q6

Explain how your renderer preserves the two correctness requirements from the
handout:

- Atomicity: all image update operations must be atomic.
- Order: updates to the same pixel must be applied in circle input order.

**Answer:**

Both invariants are satisfied **structurally, with no locks or atomics**, because
each pixel is owned by exactly one thread.

- **Atomicity.** A pixel's color is only ever read, blended, and written by its
  single owning thread, in a private `float4` register accumulator. No other
  thread touches that pixel, so there is no shared read-modify-write to make
  atomic in the first place — the critical region the starter code worried about
  simply does not exist. The pixel is written to global memory once, at the end.

- **Order.** The owning thread blends circles in strictly increasing circle
  index. Within a batch, the exclusive scan compacts hits in linear-thread-index
  order, and thread `i` corresponds to circle `batchStart + i`, so `hitList` is
  in ascending index order. Batches are processed in ascending order, and the
  register accumulator carries across batches. Therefore every pixel applies its
  contributions in exactly circle-input order, matching the sequential reference.

The handout notes that order only matters for circles touching the *same* pixel;
circles touching different pixels are independent. Owning each pixel by one
thread makes both the "same pixel" ordering and the atomicity automatic, which is
why no synchronization between threads is needed for correctness of the blend
itself (the `__syncthreads()` in Q3 only coordinate the shared-memory hit-list
construction, not the image updates).

---

## Extra Credit

### Q1

If you implemented scan using an approach competitive with Thrust, describe it
and compare against the Thrust implementation.

**Answer:**

The scan was tuned in three iterations, each guided by profiling on the same
`g5g.xlarge` (Nsight Systems for kernel counts, Nsight Compute for DRAM traffic).
Times are `Student GPU time` vs `Thrust GPU time`, 40M random ints:

| version | 40M time | vs Thrust |
|---|---:|---:|
| naive (one kernel per tree level) | 33.0 ms | 18.5x |
| block scan (shared-memory, 3-phase) | 5.84 ms | 3.3x |
| + drop redundant copy + conflict-free banks | 3.07 ms | **1.7x** |
| Thrust (CUB single-pass look-back) | 1.81 ms | 1x |

**What profiling showed, and what each fix did.** Nsight Systems confirmed Thrust
runs a single main `cub::DeviceScanKernel` using a `ScanTileState` -- i.e. the
decoupled look-back single-pass scan -- versus the 5 kernels per scan here. Nsight
Compute then decomposed the (then 3.3x) time gap of the block-scan version into
two independent factors that multiply almost exactly to the observed ratio:

- **Traffic 2.16x.** This implementation moved 771 MB of DRAM vs Thrust's 357 MB (~1.1x the 2N
  minimum). Thrust is genuinely single-pass; this implementation was multi-pass.
- **Achieved bandwidth 1.54x.** Thrust sustained 203 GB/s (63% of the T4G's
  ~320 GB/s peak) vs 132 GB/s (41%) here. The peak is fixed; the *achieved*
  fraction is not -- this kernel stalls global memory during the in-shared-memory
  tree phase and runs only 2 elements/thread, so it keeps the bus less busy.

Two of those findings became fixes (the third iteration above): (1) the original
code did a device-to-device `cudaMemcpy` of the whole array before scanning --
2N of pure waste inside the timed region -- which was removed by reading the input
and writing the result directly; (2) adding `CONFLICT_FREE_OFFSET` padding to the
Blelloch shared-memory indices removed bank conflicts and lifted achieved
bandwidth. Together these roughly halved the time (5.84 -> 3.07 ms) and closed the
gap to 1.7x, still at full marks on the checker.

**Why the tuning stopped at 1.7x.** Closing the rest means matching Thrust's traffic (2N),
which requires the decoupled look-back single-pass algorithm: a global per-tile
state array, cross-block publish/look-back with memory fences and atomic status
flags, and careful forward-progress guarantees to avoid deadlock. That is the
core of CUB; its difficulty is concurrency correctness (nondeterministic, load-
dependent bugs), not line count, and the payoff here is only the last ~1.7x on a
warm-up part that is already full marks. So this implementation is competitive
(within a small constant factor of a heavily optimized library, down from more
than an order of magnitude) rather than equal.

---

### Q2

If your renderer achieves significantly greater performance than required,
explain the approach thoroughly.

**Answer:**

The renderer is faster than the reference on every circle-heavy scene, by a wide
margin on several: snowsingle ~3.0x, micro2M ~3.3x, rand1M ~2.9x, rand100k ~1.7x,
rand10k ~1.6x, pattern ~1.4x (rgb and biglittle are at their respective floors).

Two design choices account for this, both validated by profiling (full detail in
Part 3 Q4/Q5):

1. **Tiling + per-tile culling** turns the per-pixel cost from O(N) circles into
   O(circles touching the tile). Nsight Compute confirmed the bottleneck was
   compute/L1 (iteration count), not DRAM, so cutting iterations is exactly the
   right lever; this alone reached parity with the reference.

2. **Largest legal tile (32x32 = 1024 threads).** The per-batch overhead (one
   shared-memory scan + four `__syncthreads()`) is paid `N / SCAN_BLOCK_DIM`
   times. Re-profiling the 16x16 version showed it was stalled on this overhead
   (no pipe saturated on sparse scenes). A monotonic sweep over 8/16/32 confirmed
   bigger is better; 32x32 minimizes the batch count and gave the ~3x speedups
   above (e.g. micro2M 502 -> 145 ms). 32 is the maximum because the shared-memory
   scan caps at 1024 threads.

The key methodological point is that none of this was guessed: the optimization
target (reduce iterations, then reduce batch overhead) was read off Nsight
Compute counters, and the tile size was chosen by an empirical sweep, not a hunch.

