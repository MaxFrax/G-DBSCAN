#include <stdio.h>
#include <thrust/scan.h>

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

__global__ void compute_degrees(float* dataset, int d, int n, int* degrees, float squaredThreshold) {
	extern __shared__ float coordinates[];

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int degree = 0;

	if (tid >= n)
		return;

	// 1. Load in shared memory the coordinates assigned to the current thread
    // At this stage, a nice performance boots would be assigning the unused shared memory to the L1 cache
	for (int i = 0; i < d; i++) {
		coordinates[d * threadIdx.x + i] = dataset[i * n + tid];
	}

    // 2. Compare the current thread coordinates againts all the points in device memory
    // probably memory metrics will be a nightmare.
    // Can we use shared memory somehow to optimize the accesses?
	for (int item = 0; item < n; item++) {
		float sum = 0;
		for (int dim = 0; dim < d; dim++) {
			sum += powf(coordinates[d * threadIdx.x + dim] - dataset[dim * n + item], 2);
		}

		if (sum < squaredThreshold) {
			degree += 1;
		}
	}

    // 3. Store the computed degree in degrees
	degrees[tid] = degree;
}

__global__ void compute_adjacency_list(float* dataset, int d, int n, int* degreeArray, int* adjIndexArray, int* adjList, float squaredThreshold) {
	extern __shared__ float coordinates[];

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int adjIndex, degree, foundNeighbours = 0;

	if (tid >= n)
		return;

	// 1. Load in shared memory the coordinates assigned to the current thread
    // At this stage, a nice performance boots would be assigning the unused shared memory to the L1 cache
	adjIndex = adjIndexArray[tid];
	degree = degreeArray[tid];
	for (int i = 0; i < d; i++) {
		coordinates[d * threadIdx.x + i] = dataset[i * n + tid];
	}

    // 2. Compare the current thread coordinates againts all the points in device memory
    // probably memory metrics will be a nightmare.
    // Can we use shared memory somehow to optimize the accesses?
	for (int item = 0; item < n; item++) {

		if(foundNeighbours >= degree){
			return;
		}

		float sum = 0;
		for (int dim = 0; dim < d; dim++) {
			sum += powf(coordinates[d * threadIdx.x + dim] - dataset[dim * n + item], 2);
		}

		if (sum < squaredThreshold) {
			// 3. Store the adjacent node and increment adj ix
			adjList[adjIndex + foundNeighbours] = tid;
			foundNeighbours++;
		}
	}
}

int main()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);


	int n = 1500;
	int d = 2;

	float* dataset;
	int* degrees;
	int* adjIndex;
	int *adjList;

    // 1. Alloc d n-dimensional arrays in unified memory.
    // One for each data dimension    
	CHECK(cudaMallocManaged(&dataset, d * n * sizeof(float)));
	CHECK(cudaMallocManaged(&degrees, n * sizeof(int)));
	CHECK(cudaMallocManaged(&adjIndex, n * sizeof(int)));

	for (int i = 0; i < d * n ; i++) {
        // 2. Fill the arrays with random data
		//dataset[i] = rand() / float(RAND_MAX) * 20.f - 10.f;
		dataset[i] = (i % 20) - 10;
	}

    // 3. Invoke kernel
    // smem size / (d * sizeof(float)) = number of threads per block
	printf("Phase 1: Degrees\n");
    int threadsPerBlock = min((int)(prop.sharedMemPerBlock / (d * sizeof(float))), prop.maxThreadsPerBlock);
    printf("Threads per block %d\n", threadsPerBlock);
	int blocksPerGrid = (n / threadsPerBlock) + 1;
	printf("Blocks per grid %d\n", blocksPerGrid);
	printf("Alloc smem %lu bytes of %lu bytes\n", threadsPerBlock * d * sizeof(float), prop.sharedMemPerBlock);
	compute_degrees << <blocksPerGrid, threadsPerBlock, threadsPerBlock * d * sizeof(float) >> > (dataset, d, n, degrees, .1f * .1f);

	cudaDeviceSynchronize();

	for(int i = 0; i < n; i++) {
		printf("%d\t", degrees[i]);
	}
	printf("\n");

	// 4. Create the indexes pointing to the adjacency list with a prefix sum
	printf("Phase 2: Adj List indexes\n");
	thrust::exclusive_scan(degrees, degrees + n, adjIndex); 

	for(int i = 0; i < n; i++) {
		printf("%d\t", adjIndex[i]);
	}

	// 5. Compute adjacency list

	// 6. Build the adjacency list
	int adjListSize = adjIndex[n-1] + degrees[n-1];
	CHECK(cudaMallocManaged(&adjList, adjListSize * sizeof(int)));

	printf("\nPhase 3: Adjacency list\n");
	compute_adjacency_list << <blocksPerGrid, threadsPerBlock, threadsPerBlock * d * sizeof(float) >> > (dataset, d, n, degrees, adjIndex, adjList, .1f * .1f);

	cudaDeviceSynchronize();

	for(int i = 0; i < adjListSize; i++) {
		printf("%d\t", adjList[i]);
	}

	// 7. Free memory
	CHECK(cudaFree(dataset));
	CHECK(cudaFree(degrees));
	CHECK(cudaFree(adjIndex));
	CHECK(cudaFree(adjList));

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
