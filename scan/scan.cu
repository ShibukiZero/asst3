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

__global__ void block_scan_kernel(int* data, int n, int* block_sums);
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

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
// Recursively exclusive-scans `n` elements of the device array `data` in place.
// Phase 1 scans each ELEMENTS_PER_BLOCK-sized chunk locally in shared memory and
// emits that chunk's total; phase 2 scans the chunk totals (recursing when there
// is more than one chunk); phase 3 adds each chunk's offset back. Global-memory
// traffic is only a couple of passes -- unlike the naive one-kernel-per-tree-level
// version that streams the whole array through global memory ~2*log2(N) times.
static void scan_in_place(int* data, int n) {
    int num_blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    int* block_sums = nullptr;
    cudaMalloc((void**)&block_sums, sizeof(int) * num_blocks);

    size_t shared_bytes = ELEMENTS_PER_BLOCK * sizeof(int);

    // Phase 1: each block exclusive-scans its own chunk and writes the chunk
    // total into block_sums.
    block_scan_kernel<<<num_blocks, THREADS_PER_BLOCK, shared_bytes>>>(data, n, block_sums);

    if (num_blocks > 1) {
        // Phase 2: exclusive-scan the chunk totals so block_sums[i] becomes the
        // offset that chunk i's elements need.
        scan_in_place(block_sums, num_blocks);
        // Phase 3: add each chunk's offset back into its elements.
        add_block_offsets_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(data, n, block_sums);
    }

    cudaFree(block_sums);
}

void exclusive_scan(int* input, int N, int* result)
{
    if (N <= 0) {
        return;
    }

    // The scan runs in place on `result`; seed it with the input. No power-of-two
    // padding is needed because the block scan bounds-checks its tail.
    cudaMemcpy(result, input, sizeof(int) * N, cudaMemcpyDeviceToDevice);
    scan_in_place(result, N);
}

// Work-efficient (Blelloch) exclusive scan of one chunk, done entirely in shared
// memory. Each thread loads two elements; tail elements past `n` load 0. The
// chunk's total sum is written to block_sums[blockIdx.x].
__global__ void block_scan_kernel(int* data, int n, int* block_sums) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int base = blockIdx.x * ELEMENTS_PER_BLOCK;
    int ai = tid;
    int bi = tid + THREADS_PER_BLOCK;   // second half of the chunk

    temp[ai] = (base + ai < n) ? data[base + ai] : 0;
    temp[bi] = (base + bi < n) ? data[base + bi] : 0;

    int offset = 1;

    // upsweep / reduce: build partial sums up the tree
    for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int x = offset * (2 * tid + 1) - 1;
            int y = offset * (2 * tid + 2) - 1;
            temp[y] += temp[x];
        }
        offset <<= 1;
    }

    // stash the chunk total, then clear the root before the downsweep
    if (tid == 0) {
        block_sums[blockIdx.x] = temp[ELEMENTS_PER_BLOCK - 1];
        temp[ELEMENTS_PER_BLOCK - 1] = 0;
    }

    // downsweep: distribute partial sums back down to exclusive prefix sums
    for (int d = 1; d < ELEMENTS_PER_BLOCK; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int x = offset * (2 * tid + 1) - 1;
            int y = offset * (2 * tid + 2) - 1;
            int t = temp[x];
            temp[x] = temp[y];
            temp[y] += t;
        }
    }
    __syncthreads();

    if (base + ai < n) data[base + ai] = temp[ai];
    if (base + bi < n) data[base + bi] = temp[bi];
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
