extern "C" __global__ void compute_degrees(float* dataset, int d, int n, int* degrees, float squaredThreshold) {
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

extern "C"  __global__ void compute_adjacency_list(float* dataset, int d, int n, int* degreeArray, int* adjIndexArray, int* adjList, float squaredThreshold) {
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
			adjList[adjIndex + foundNeighbours] = item;
			foundNeighbours++;
		}
	}
}