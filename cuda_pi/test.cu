#include <stdio.h>

__global__ void c_hello() {
	printf("Hello World from the GPU! (ThrIndex:%d)\n", threadIdx.x);
}

int main() {
	c_hello <<<1,10>>>();
	cudaDeviceSynchronize();
	//Needed for output.
	return 0;
}