#include <stdio.h>
#include "matrix.h"

#define debug TRUE

__global__ void c_hello() {
	printf("Hello World from the GPU! (ThrIndex:%d)\n", threadIdx.x);
}

int main(int argc, char *argv[]) {
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

	c_hello <<<1,10>>>();
	cudaDeviceSynchronize();
	//Needed for output.
	return 0;
}