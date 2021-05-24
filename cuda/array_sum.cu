#include <stdio.h>
#include <math.h>

// CUDA kernel to add elements of two arrays
__global__ void add(float* x, float* y)
{
   y[threadIdx.x] = x[threadIdx.x] + y[threadIdx.x];
}

int main(void)
{
   //Variables
   int N = 1024;
   float* x, * y;
   int size = sizeof(float) * N;

   // Allocate Unified Memory -- accessible from CPU or GPU
   cudaMallocManaged(&x, size);
   cudaMallocManaged(&y, size);

   //Initialize x and y arrays on the host using unified pointers.
   for (int i = 0; i < N; i++) {
      x[i] = 1.0;
      y[i] = 2.0;
   }

   //Launch kernel on N elements on the GPU
   add <<<1, N >>> (x, y);

   // Wait for GPU to finish before accessing on host
   cudaDeviceSynchronize();

   // Check for errors (all values should be 3.0)
   float maxError = 0.0;
   for (int i = 0; i < N; i++)
      maxError = (float)fmax(maxError, fabs(y[i] - 3.0));
   printf("Max error: %lf\n", maxError);

   // Free cuda memory
   cudaFree(x);
   cudaFree(y);

   return 0;
}
