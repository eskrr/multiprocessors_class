#include <stdio.h>
#include "matrix.h"

#define DEBUG TRUE

void printDeviceInfo() {
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for(inti = 0; i < nDevices; i++) {
		cudaDeviceProp prop;c
		udaGetDeviceProperties(&prop, i);
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

__global__ void c_hello() {
	printf("Hello World from the GPU! (ThrIndex:%d)\n", threadIdx.x);
}

int main(int argc, char *argv[]) {
	if (DEBUG)
		printDeviceInfo();

	if (!verifyArgs(argc))
		return false;

	MATRIX *mA, *mB;
	if (!initializeInputMatrixes(argc, argv, &mA, &mB, DEBUG))
		return -1;

	MATRIX* mC;

	if  ((mC = initializeOutputMatrix(*mA, *mB)) == NULL) {
		printf("Error allocating output matrix C.\n");
		return -1;
	}

	printf("Answer size: %d, %d\n", mC->rows, mC->cols);

	c_hello <<<10 ,10>>>();
	cudaDeviceSynchronize();
	//Needed for output.
	return 0;
}