#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include "matrix.h"

#define DEBUG true
#define MAX_THREADS 2014
#define MAX_BLOCKS 1024

clock_t start, end;

void printDeviceInfo() {
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	int i = 0;
	for(i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Maximum number of 32-bit registers: %d\n", prop.regsPerBlock);
		printf("  Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("  Maximum block dimension: [%d,%d,%d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("  Maximum grid size: [%d,%d,%d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	}
}

__global__ void calculateMatrixCuda(int *workPerThread, MATRIX* mA, MATRIX* mB, MATRIX* mC) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate index for each thread
	int startPos = idx * *workPerThread;
	int endPos = startPos + *workPerThread;

	printf("(ThreadId: %d, WorkPerThread: %d)\n", idx, *workPerThread);
	printf("(start: %d, end: %d)\n", startPos, endPos);
	printf("mA: rows: %d, cols: %d\n", mA->rows, mA->cols);
	printf("mB: rows: %d, cols: %d\n", mB->rows, mB->cols);
	printf("mC: rows: %d, cols: %d\n", mC->rows, mC->cols);
}

int main(int argc, char *argv[]) {
	if (DEBUG)
		printDeviceInfo();

	if (!verifyArgs(argc))
		return false;

	MATRIX *mA, *mB;
	if (!initializeInputMatrixes(argc, argv, &mA, &mB, DEBUG, true))
		printf("Error allocating input matrixes.\n");
		return -1;

	printf("Inpit Matrix allocated\n");

	MATRIX* mC;

	if  ((mC = initializeOutputMatrix(*mA, *mB, true)) == NULL) {
		printf("Error allocating output matrix C.\n");
		return -1;
	}

	printf("Que pedo?\n");

	int *workPerThread;
	cudaMallocManaged(&workPerThread, sizeof(int));
	printf("Que pedo2?\n");

	int totalBlocks = mC->rows < MAX_BLOCKS ?  mC->rows : MAX_BLOCKS;
	int totalRows = mC->cols < MAX_THREADS ?  mC->cols : MAX_THREADS;
	printf("Que pedo4?\n");

	int totalWork = mC->rows * mC->cols;
	*workPerThread = totalWork / (totalBlocks * totalRows);

	printf("totalBlocks: %d, totalRows: %d\n", totalBlocks, totalRows);

	start = clock();
	calculateMatrixCuda <<<totalBlocks, totalRows>>> (workPerThread, mA, mB, mC);
	cudaDeviceSynchronize();
	end = clock();

 	// double totalTime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total time taken by CPU: %lf\n", end - start); 

	printf("Verifying matrix... \n");
	//Needed for output.
	return 0;
}