#include <stdio.h>

__global__ void compute_degrees (float ** dataset, int d, int n, int * degrees, float threshold) {
    // 1. Load in shared memory the coordinates assigned to the current thread
    // At this stage, a nice performance boots would be assigning the unused shared memory to the L1 cache

    // 2. Compare the current thread coordinates againts all the points in device memory
    // probably memory metrics will be a nightmare.
    // Can we use shared memory somehow to optimize the accesses?

    // 3. Store the computed degree in degrees
}

int main(void) {
    int n = 100_000;
    int d = 2;

    // 1. Alloc d n-dimensional arrays in unified memory.
    // One for each data dimension

    // 2. Fill the arrays with random data

    // 3. How do we compute grid and block size in a smart way?

    // 4. Invoke kernel

    cudaDeviceReset();
    
    return 0;
}
