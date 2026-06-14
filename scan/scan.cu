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

__global__ void upsweep_kernel(int* data, int two_d, int numTasks);
__global__ void downsweep_kernel(int* data, int two_d, int numTasks);
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
void exclusive_scan(int* input, int N, int* result)
{

    // This runs on the CPU and drives the work-efficient parallel scan by
    // launching one kernel per upsweep/downsweep level. Each level launches
    // exactly one thread per active task (not one per element), so the total
    // work stays O(N) rather than O(N log N).
    if (N <= 0) {
        return;
    }

    int rounded_N = nextPow2(N);

    // Scan runs in-place on `result`: seed it with the input, then zero-pad
    // the tail out to the next power of two so the tree algorithm is exact.
    cudaMemcpy(result, input, sizeof(int) * N, cudaMemcpyDeviceToDevice);
    if (rounded_N > N) {
        cudaMemset(result + N, 0, sizeof(int) * (rounded_N - N));
    }

    // Upsweep (reduce): each level sums pairs up the tree. At distance two_d
    // there are rounded_N / (2*two_d) active tasks, so launch exactly that many.
    for (int two_d = 1; two_d <= rounded_N / 2; two_d *= 2) {
        int two_dplus1 = 2 * two_d;
        int numTasks = rounded_N / two_dplus1;
        int blocks = (numTasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        upsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(result, two_d, numTasks);
    }

    // Clear the root, then downsweep distributes the partial sums back down
    // the tree to produce the exclusive prefix sums.
    cudaMemset(result + rounded_N - 1, 0, sizeof(int));

    for (int two_d = rounded_N / 2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2 * two_d;
        int numTasks = rounded_N / two_dplus1;
        int blocks = (numTasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        downsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(result, two_d, numTasks);
    }
}

__global__ void upsweep_kernel(int* data, int two_d, int numTasks) {
    int task = blockIdx.x * blockDim.x + threadIdx.x;

    if (task < numTasks) {
        int two_dplus1 = 2 * two_d;
        int i = task * two_dplus1;
        data[i + two_dplus1 - 1] += data[i + two_d - 1];
    }
}

__global__ void downsweep_kernel(int* data, int two_d, int numTasks) {
    int task = blockIdx.x * blockDim.x + threadIdx.x;

    if (task < numTasks) {
        int two_dplus1 = 2 * two_d;
        int i = task * two_dplus1;
        int t = data[i+two_d-1];
        data[i+two_d-1] = data[i+two_dplus1-1];
        data[i+two_dplus1-1] += t;
    }
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
