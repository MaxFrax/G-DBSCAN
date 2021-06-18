#include <stdio.h>

#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                            \
    }                                                                       \
}

__global__ void compute_degrees(float** dataset, int d, int n, int* degrees, float squaredThreshold) {
	extern __shared__ float coordinates[];

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int degree = 0;

	if (tid >= n)
		return;

	// 1. Load in shared memory the coordinates assigned to the current thread
    // At this stage, a nice performance boots would be assigning the unused shared memory to the L1 cache
	for (int i = 0; i < d; i++) {
		coordinates[d * threadIdx.x + i] = dataset[i][tid];
	}

    // 2. Compare the current thread coordinates againts all the points in device memory
    // probably memory metrics will be a nightmare.
    // Can we use shared memory somehow to optimize the accesses?
	for (int item = 0; item < n; item++) {
		float sum = 0;
		for (int dim = 0; dim < d; dim++) {
			sum += powf(coordinates[d * threadIdx.x + dim] - dataset[dim][item], 2);
		}

		if (sum < squaredThreshold) {
			degree += 1;
		}
	}

    // 3. Store the computed degree in degrees
	degrees[tid] = degree;
}

int main()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);


	int n = 100000;
	int d = 2;

	float** dimensions;
	int* degrees;

    // 1. Alloc d n-dimensional arrays in unified memory.
    // One for each data dimension    
	CHECK(cudaMallocManaged(&dimensions, d * sizeof(float*)));
	CHECK(cudaMallocManaged(&degrees, n * sizeof(int)));

	for (int i = 0; i < d; i++) {
		float* j;
		CHECK(cudaMallocManaged(&j, n * sizeof(float)));

        // 2. Fill the arrays with random data
		for (int k = 0; k < n; k++) {
			j[k] = rand() / float(RAND_MAX) * 20.f - 10.f;
		}

		dimensions[i] = j;
	}

    // 3. Invoke kernel
    // smem size / (d * sizeof(float)) = number of threads per block
    int threadsPerBlock = min((int)(prop.sharedMemPerBlock / (d * sizeof(float))), prop.maxThreadsPerBlock);
    printf("Threads per block %d\n", threadsPerBlock);
	int blocksPerGrid = (n / threadsPerBlock) + 1;
	printf("Blocks per grid %d\n", blocksPerGrid);
	printf("Alloc smem %lu bytes of %lu bytes\n", threadsPerBlock * d * sizeof(float), prop.sharedMemPerBlock);
	compute_degrees << <blocksPerGrid, threadsPerBlock, threadsPerBlock * d * sizeof(float) >> > (dimensions, d, n, degrees, .1f * .1f);

	cudaDeviceSynchronize();

	for(int i = 0; i < n; i++) {
		printf("%d\t", degrees[i]);
	}

	// 4. Free memory
	for (int i = 0; i < d; i++) {
        printf("Freeing %d\n", i);
		CHECK(cudaFree(dimensions[i]));
	}

	CHECK(cudaFree(dimensions));
	CHECK(cudaFree(degrees));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}