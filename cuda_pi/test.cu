#include<stdio.h>
__global__ voidc_hello() {
	printf("Hello World from the GPU!\n");
}

int main() {
	c_hello <<<1,10>>>();
	cudaDeviceSynchronize();
	//Needed for output.
	return 0;
}