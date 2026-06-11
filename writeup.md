# Assignment 3: A Simple CUDA Renderer

**Name(s):**
**SUNet ID(s):**
**Machine used for measurements:**

---

## Part 1: CUDA Warm-Up 1: SAXPY

### Q1

What performance do you observe compared to the sequential CPU-based
implementation of SAXPY (recall your results from saxpy on Program 5 from
Assignment 1)?

**Answer:**



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



---

## Part 2: CUDA Warm-Up 2: Parallel Prefix-Sum

### Q1

Implement `exclusive_scan` in `scan/scan.cu` using the iterative upsweep /
downsweep parallel prefix-sum algorithm described in the handout.

**Answer:**



---

### Q2

Implement `find_repeats` in `scan/scan.cu`. Given an input array `A`, it should
return the list of all indices `i` for which `A[i] == A[i+1]`. The implementation
should use one or more calls to `exclusive_scan`.

**Answer:**



---

### Q3

Report the correctness and performance results from:

```bash
./checker.py scan
./checker.py find_repeats
```

**Answer:**

| Test | Element Count | Ref Time | Student Time | Score |
|---|---:|---:|---:|---:|
| scan |  |  |  |  |
| scan |  |  |  |  |
| scan |  |  |  |  |
| scan |  |  |  |  |
| find_repeats |  |  |  |  |
| find_repeats |  |  |  |  |
| find_repeats |  |  |  |  |
| find_repeats |  |  |  |  |

Total scan score:
Total find_repeats score:

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
