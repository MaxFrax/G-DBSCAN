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

	// 1. Loads in shared memory the coordinates assigned to the current thread
	for (int i = 0; i < d; i++)
	{
		coordinates[d * threadIdx.x + i] = dataset[i * n + tid];
	}

	// 2. Compare the current thread coordinates againts all the points in device memory
	for (int item = 0; item < n; item++)
	{
		float sum = 0;
		// Sum the squared difference for each dimension
		for (int dim = 0; dim < d; dim++)
		{
			sum += powf(coordinates[d * threadIdx.x + dim] - dataset[dim * n + item], 2);
		}

		// Is item close enough? If so, increase the degree
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

	adjIndex = adjIndexArray[tid];
	degree = degreeArray[tid];

	// 1. Loads in shared memory the coordinates assigned to the current thread
	for (int i = 0; i < d; i++)
	{
		coordinates[d * threadIdx.x + i] = dataset[i * n + tid];
	}

	// 2. Compare the current thread coordinates againts all the points in device memory
	for (int item = 0; item < n; item++)
	{
		// If we found all the expected neighbours, the job of this kernel is done
		if (foundNeighbours >= degree)
		{
			return;
		}

		// Sum the squared difference for each dimension
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

	// 1. Is this node in the frontier? If not, job is done
	if (Fa[tid] == 0)
	{
		return;
	}

	// 2. Remove the node from the frontier and add it to the visisted
	Fa[tid] = 0;
	Xa[tid] = 1;

	int adjListBegin = adjListIx[tid];

	// Foreach neighbour
	for (int i = 0; i < degrees[tid]; i++)
	{
		int nid = adjList[adjListBegin + i];

		// If we have still to visit it, we add it to the frontier
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

	// If the node has been visited and still not have a cluster, we add it to the current one
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
		// The frontier is not empty, so let's execute a step of bfs
		kernel_bfs<<<blocksPerGrid, blocks>>>(Fa, Xa, n, degrees, adjListIx, adjList);
		cudaDeviceSynchronize();

		// Checks if the frontier is empty
		int toFind = 1;
		int *res = thrust::find(thrust::device, Fa, Fa + n, toFind);
		// If the pointer is "last", the search didn't find a 1. the frontier is empty
		if (res == Fa + n)
		{
			FaEmpty = true;
		}
	}

	// Foreach visited node (Xa == 1) which is not assigned to a cluster, assign it to currentCluster
	cluster_assignment<<<blocksPerGrid, blocks>>>(Xa, cluster, n, currentCluster);
	cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
	// We give most of the shared memory to l1 cache for each kernel
	// Notice: in case of big d in input this could cause issues
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
		printf("Missing input file");
		return 1;
	}

	FILE *fp = fopen(argv[1], "r");

	fscanf(fp, "# %d %d %f %d", &n, &d, &threshold, &MinPts);

	float *dataset;
	int *degrees;
	int *adjIndex;
	int *adjList;

	// 1. Alloc linearized dataset matrix
	CHECK(cudaMallocManaged(&dataset, d * n * sizeof(float)));
	CHECK(cudaMallocManaged(&degrees, n * sizeof(int)));
	CHECK(cudaMallocManaged(&adjIndex, n * sizeof(int)));

	// 2. Read the input file
	// A sequence of n numbers belongs to the same column
	for (int i = 0; i < d * n; i++)
	{
		float read;
		fscanf(fp, "%f", &read);

		dataset[i] = read;
	}

	fclose(fp);

	// 3. Invoke kernel compute degrees
	// We launch the maximum number of threads in a block. 
	// This means the maximum between the number of threads the shared memory can fit and the maximum number of threads the device supports.
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

	clock_t t;
	t = clock();
	// 4. Create the indexes pointing to the adjacency list with a prefix sum
	thrust::exclusive_scan(thrust::device, degrees, degrees + n, adjIndex);
	t = clock() - t;
	double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
	printf("Exclusive scan elapsed time               : %.3f (sec)\n", time_taken);

	// 5. Compute adjacency list
	// The list length is the the sum between the last index and the degree of the corresponding point
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

	CHECK(cudaEventRecord(start));

	// Foreach node
	for (int v = 0; v < n; v++)
	{
		if (cluster[v] > 0 || degrees[v] < MinPts)
			continue;

		// If the node is without a cluster and it is a core node
		bfs(&prop, Fa, Xa, v, n, cluster, nextCluster, degrees, adjIndex, adjList);
		nextCluster++;
	}

	CHECK(cudaDeviceSynchronize());
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("BFS elapsed time               : %.3f (sec)\n", milliseconds / 1000.0);


	// 7. Store the result
	fp = fopen("out.txt", "w");

	for (int i = 0; i < n; i++)
	{
		fprintf(fp, "%d\n", cluster[i]);
	}

	fclose(fp);

	// 8. Free memory
	CHECK(cudaFree(dataset));
	CHECK(cudaFree(degrees));
	CHECK(cudaFree(adjIndex));
	CHECK(cudaFree(adjList));
	CHECK(cudaFree(Xa));
	CHECK(cudaFree(Fa));
	CHECK(cudaFree(cluster));
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
