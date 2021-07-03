#include <stdio.h>
#include <thrust/scan.h>
#include <time.h>

#define CHECK(call)                                                            \
	{                                                                          \
		const cudaError_t error = call;                                        \
		if (error != cudaSuccess)                                              \
		{                                                                      \
			printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
			printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
			exit(1);                                                           \
		}                                                                      \
	}

__global__ void compute_degrees(float *dataset, int d, int n, int *degrees, float squaredThreshold)
{
	extern __shared__ float coordinates[];

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int degree = 0;

	if (tid >= n)
		return;

	// 1. Load in shared memory the coordinates assigned to the current thread
	// At this stage, a nice performance boost would be assigning the unused shared memory to the L1 cache
	for (int i = 0; i < d; i++)
	{
		coordinates[d * threadIdx.x + i] = dataset[i * n + tid];
	}

	// 2. Compare the current thread coordinates againts all the points in device memory
	// probably memory metrics will be a nightmare.
	// Can we use shared memory somehow to optimize the accesses?
	for (int item = 0; item < n; item++)
	{

		float sum = 0;
		for (int dim = 0; dim < d; dim++)
		{
			sum += powf(coordinates[d * threadIdx.x + dim] - dataset[dim * n + item], 2);
		}

		if (sum <= squaredThreshold)
		{
			degree += 1;
		}
	}

	// 3. Store the computed degree in degrees
	degrees[tid] = degree;
}

__global__ void compute_adjacency_list(float *dataset, int d, int n, int *degreeArray, int *adjIndexArray, int *adjList, float squaredThreshold)
{
	extern __shared__ float coordinates[];

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int adjIndex, degree, foundNeighbours = 0;

	if (tid >= n)
		return;

	// 1. Load in shared memory the coordinates assigned to the current thread
	// At this stage, a nice performance boost would be assigning the unused shared memory to the L1 cache
	adjIndex = adjIndexArray[tid];
	degree = degreeArray[tid];
	for (int i = 0; i < d; i++)
	{
		coordinates[d * threadIdx.x + i] = dataset[i * n + tid];
	}

	// 2. Compare the current thread coordinates againts all the points in device memory
	// probably memory metrics will be a nightmare.
	// Can we use shared memory somehow to optimize the accesses?
	for (int item = 0; item < n; item++)
	{

		if (foundNeighbours >= degree)
		{
			return;
		}

		float sum = 0;
		for (int dim = 0; dim < d; dim++)
		{
			sum += powf(coordinates[d * threadIdx.x + dim] - dataset[dim * n + item], 2);
		}

		if (sum <= squaredThreshold)
		{
			// 3. Store the adjacent node and increment adj ix
			adjList[adjIndex + foundNeighbours] = item;
			foundNeighbours++;
		}
	}
}

__global__ void kernel_bfs(int *Fa, int *Xa, int n, int *degrees, int *adjListIx, int *adjList)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n)
		return;

	// This could cause warp divergency
	if (Fa[tid] == 0)
	{
		return;
	}

	Fa[tid] = 0;
	Xa[tid] = 1;

	int adjListBegin = adjListIx[tid];

	// Foreach neighbour
	for (int i = 0; i < degrees[tid]; i++)
	{
		int nid = adjList[adjListBegin + i];

		if (Xa[nid] == 0)
		{
			Fa[nid] = 1;
		}
	}
}

__global__ void cluster_assignment(int *Xa, int *cluster, int n, int currentCluster)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n)
		return;

	if (Xa[tid] == 1 && cluster[tid] == 0)
	{
		cluster[tid] = currentCluster;
	}
}

__host__ void bfs(cudaDeviceProp *prop, int *Fa, int *Xa, int v, int n, int *cluster, int currentCluster, int *degrees, int *adjListIx, int *adjList)
{

	int blocks = prop->maxThreadsPerBlock;

	int blocksPerGrid = (n / blocks) + 1;

	bool FaEmpty = false;
	Fa[v] = 1;

	while (!FaEmpty)
	{

		kernel_bfs<<<blocksPerGrid, blocks>>>(Fa, Xa, n, degrees, adjListIx, adjList);
		cudaDeviceSynchronize();

		// Checks if the frontier is empty
		int toFind = 1;
		int *res = thrust::find(thrust::device, Fa, Fa + n, toFind);
		// If the pointer is "last", the search failed
		if (res == Fa + n)
		{
			break;
		}
	}

	// Foreach visited node (Xa == 1) which is not assigned to a cluster, assign it to currentCluster
	cluster_assignment<<<blocksPerGrid, blocks>>>(Xa, cluster, n, currentCluster);
	cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
	cudaFuncSetCacheConfig(compute_degrees, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(compute_adjacency_list, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(cluster_assignment, cudaFuncCachePreferL1);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int n = -1;
	int d = -1;
	float threshold;
	int MinPts;

	if (argc != 2)
	{
		return 1;
	}

	FILE *fp = fopen(argv[1], "r");

	fscanf(fp, "# %d %d %f %d", &n, &d, &threshold, &MinPts);

	float *dataset;
	int *degrees;
	int *adjIndex;
	int *adjList;

	// 1. Alloc d n-dimensional arrays in unified memory.
	// One for each data dimension
	CHECK(cudaMallocManaged(&dataset, d * n * sizeof(float)));
	CHECK(cudaMallocManaged(&degrees, n * sizeof(int)));
	CHECK(cudaMallocManaged(&adjIndex, n * sizeof(int)));

	for (int i = 0; i < d * n; i++)
	{
		// 2. Fill the arrays with random data
		float read;
		fscanf(fp, "%f", &read);

		dataset[i] = read;
	}

	fclose(fp);

	// 3. Invoke kernel
	int blocks = min((int)(prop.sharedMemPerBlock / (d * sizeof(float))), prop.maxThreadsPerBlock);
	int blocksPerGrid = (n / blocks) + 1;

	float milliseconds;
	CHECK(cudaEventRecord(start));
	compute_degrees<<<blocksPerGrid, blocks, blocks * d * sizeof(float)>>>(dataset, d, n, degrees, threshold * threshold);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("Compute degrees elapsed time               : %.3f (sec)\n", milliseconds / 1000.0);

	// 4. Create the indexes pointing to the adjacency list with a prefix sum
	CHECK(cudaEventRecord(start));

	clock_t t;
	t = clock();
	thrust::exclusive_scan(thrust::device, degrees, degrees + n, adjIndex);
	t = clock() - t;
	double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
	printf("Exclusive scan elapsed time               : %.3f (sec)\n", time_taken);

	// 5. Compute adjacency list
	int adjListSize = adjIndex[n - 1] + degrees[n - 1];
	CHECK(cudaMallocManaged(&adjList, adjListSize * sizeof(int)));

	CHECK(cudaEventRecord(start));

	compute_adjacency_list<<<blocksPerGrid, blocks, blocks * d * sizeof(float)>>>(dataset, d, n, degrees, adjIndex, adjList, threshold * threshold);

	CHECK(cudaDeviceSynchronize());
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("Compute adj list elapsed time               : %.3f (sec)\n", milliseconds / 1000.0);

	cudaDeviceSynchronize();

	// 6. BFS
	int *Xa, *Fa, *cluster;
	int nextCluster = 1;

	CHECK(cudaMallocManaged(&Xa, n * sizeof(int)));
	CHECK(cudaMallocManaged(&Fa, n * sizeof(int)));
	CHECK(cudaMallocManaged(&cluster, n * sizeof(int)));

	CHECK(cudaMemset(Xa, 0, n * sizeof(int)));
	CHECK(cudaMemset(Fa, 0, n * sizeof(int)));

	// Foreach core node:
	// bfs if not already assigned to a cluster
	CHECK(cudaEventRecord(start));

	for (int v = 0; v < n; v++)
	{
		if (cluster[v] > 0 || degrees[v] < MinPts)
			continue;

		bfs(&prop, Fa, Xa, v, n, cluster, nextCluster, degrees, adjIndex, adjList);
		nextCluster++;
	}

	CHECK(cudaDeviceSynchronize());
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("BFS elapsed time               : %.3f (sec)\n", milliseconds / 1000.0);

	fp = fopen("out.txt", "w");

	for (int i = 0; i < n; i++)
	{
		fprintf(fp, "%d\n", cluster[i]);
	}

	fclose(fp);

	// 7. Free memory
	CHECK(cudaFree(dataset));
	CHECK(cudaFree(degrees));
	CHECK(cudaFree(adjIndex));
	CHECK(cudaFree(adjList));
	CHECK(cudaFree(Xa));
	CHECK(cudaFree(Fa));
	CHECK(cudaFree(cluster));

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
