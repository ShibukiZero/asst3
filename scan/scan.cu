#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256
// Each block of THREADS_PER_BLOCK threads scans 2 elements per thread.
#define ELEMENTS_PER_BLOCK (2 * THREADS_PER_BLOCK)

// Pad shared indices by index/32 to spread the Blelloch tree's power-of-two
// strides across the GPU's 32 banks (avoids bank conflicts). SCAN_SMEM is the
// resulting padded footprint for one chunk.
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)
#define SCAN_SMEM (ELEMENTS_PER_BLOCK + (ELEMENTS_PER_BLOCK >> LOG_NUM_BANKS))

__global__ void block_scan_kernel(const int* in, int* out, int n, int* block_sums);
__global__ void add_block_offsets_kernel(int* data, int n, int* block_offsets);
__global__ void make_flags_kernel(int* input, int length, int* flags);
__global__ void scatter_repeats_kernel(int* flags, int* positions, int length, int* output);


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// Exclusive-scans `n` elements from `in` into `out` in three phases: each block
// scans an ELEMENTS_PER_BLOCK chunk in shared memory and emits its total; the
// chunk totals are scanned (recursively); each chunk's offset is added back.
// Touches global memory only a few times, vs ~2*log2(N) passes for a naive
// one-kernel-per-tree-level scan.
static void scan_device(const int* in, int* out, int n) {
    int num_blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    int* block_sums = nullptr;
    cudaMalloc((void**)&block_sums, sizeof(int) * num_blocks);

    size_t shared_bytes = SCAN_SMEM * sizeof(int);

    // Phase 1: scan each chunk locally (in -> out), emit chunk totals.
    block_scan_kernel<<<num_blocks, THREADS_PER_BLOCK, shared_bytes>>>(in, out, n, block_sums);

    if (num_blocks > 1) {
        // Phase 2: scan the chunk totals so block_sums[i] becomes chunk i's offset.
        scan_device(block_sums, block_sums, num_blocks);
        // Phase 3: add each chunk's offset back.
        add_block_offsets_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(out, n, block_sums);
    }

    cudaFree(block_sums);
}

void exclusive_scan(int* input, int N, int* result)
{
    if (N <= 0) {
        return;
    }
    // block scan reads `input` and writes `result` directly -- no copy, no padding.
    scan_device(input, result, N);
}

// Blelloch exclusive scan of one chunk in shared memory: read `in`, write `out`,
// emit the chunk total to block_sums. Tail past `n` loads 0; shared indices use
// CONFLICT_FREE_OFFSET padding.
__global__ void block_scan_kernel(const int* in, int* out, int n, int* block_sums) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int base = blockIdx.x * ELEMENTS_PER_BLOCK;
    int ai = tid;
    int bi = tid + THREADS_PER_BLOCK;   // second half of the chunk
    int oa = CONFLICT_FREE_OFFSET(ai);
    int ob = CONFLICT_FREE_OFFSET(bi);

    temp[ai + oa] = (base + ai < n) ? in[base + ai] : 0;
    temp[bi + ob] = (base + bi < n) ? in[base + bi] : 0;

    int offset = 1;

    // upsweep / reduce: build partial sums up the tree
    for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int x = offset * (2 * tid + 1) - 1;
            int y = offset * (2 * tid + 2) - 1;
            x += CONFLICT_FREE_OFFSET(x);
            y += CONFLICT_FREE_OFFSET(y);
            temp[y] += temp[x];
        }
        offset <<= 1;
    }

    // stash the chunk total, then clear the root before the downsweep
    if (tid == 0) {
        int last = ELEMENTS_PER_BLOCK - 1;
        last += CONFLICT_FREE_OFFSET(last);
        block_sums[blockIdx.x] = temp[last];
        temp[last] = 0;
    }

    // downsweep: distribute partial sums back down to exclusive prefix sums
    for (int d = 1; d < ELEMENTS_PER_BLOCK; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int x = offset * (2 * tid + 1) - 1;
            int y = offset * (2 * tid + 2) - 1;
            x += CONFLICT_FREE_OFFSET(x);
            y += CONFLICT_FREE_OFFSET(y);
            int t = temp[x];
            temp[x] = temp[y];
            temp[y] += t;
        }
    }
    __syncthreads();

    if (base + ai < n) out[base + ai] = temp[ai + oa];
    if (base + bi < n) out[base + bi] = temp[bi + ob];
}

// Phase 3: add chunk i's scanned offset to every element of chunk i.
__global__ void add_block_offsets_kernel(int* data, int n, int* block_offsets) {
    int base = blockIdx.x * ELEMENTS_PER_BLOCK;
    int add = block_offsets[blockIdx.x];

    int ai = base + threadIdx.x;
    int bi = base + threadIdx.x + THREADS_PER_BLOCK;
    if (ai < n) data[ai] += add;
    if (bi < n) data[bi] += add;
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // Strategy: build a 0/1 flag array where flag[i] = 1 iff input[i] ==
    // input[i+1], exclusive-scan it to get the output slot of each match, then
    // scatter the matching indices into those slots. The match count is the
    // scan's last element plus the last flag.
    if (length <= 1) {
        return 0;
    }

    int flag_length = length - 1;
    int rounded_length = nextPow2(flag_length);

    int* flags;
    int* positions;
    cudaMalloc((void**)&flags, sizeof(int) * rounded_length);
    cudaMalloc((void**)&positions, sizeof(int) * rounded_length);

    int blocks = (flag_length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // flags[i] = 1 if input[i] == input[i+1], else 0
    make_flags_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_input, flag_length, flags);

    // positions[i] = number of matches strictly before i = output slot for match i
    exclusive_scan(flags, flag_length, positions);

    // write each matching index i into output[positions[i]]
    scatter_repeats_kernel<<<blocks, THREADS_PER_BLOCK>>>(flags, positions, flag_length, device_output);

    // total matches = exclusive-scan total = positions[last] + flags[last]
    int last_position;
    int last_flag;
    cudaMemcpy(&last_position, positions + flag_length - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_flag, flags + flag_length - 1, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(flags);
    cudaFree(positions);

    return last_position + last_flag;
}

__global__ void make_flags_kernel(int* input, int length, int* flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length){
        flags[idx] = (input[idx] == input[idx + 1]) ? 1 : 0;
    }
}

__global__ void scatter_repeats_kernel(int* flags, int* positions, int length, int* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length && flags[idx] == 1) {
        output[positions[idx]] = idx;
    }
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
