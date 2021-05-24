#include <stdio.h>

#define STEPS  2000027648
#define BLOCKS   512
#define THREADS 128

int threadidx;
double pi = 0;

// Kernel
__global__ void pi_calculation(double* sum, int nsteps, double base, int nthreads, int nblocks)
{
   int i;
   double x;
   int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate index for each thread
   for (i = idx; i < nsteps; i += nthreads * nblocks)
   {
      x = (i + 0.5) * base;
      sum[idx] += 4.0 / (1.0 + x * x);  //Save result to device memory
   }
}

int main(void)
{
   dim3 dimGrid(BLOCKS, 1, 1);  // Grid dimensions
   dim3 dimBlock(THREADS, 1, 1);  // Block dimensions
   double* h_sum, * d_sum;  // Pointer to host & device arrays
   double base = 1.0 / STEPS;  // base size
   size_t size = BLOCKS * THREADS * sizeof(double);  //Array memory size

   // Memory allocation
   h_sum = (double*)malloc(size);  //  Allocate array on host
   cudaMalloc((void**)&d_sum, size);  // Allocate array on device

   // Initialize array in device to 0
   cudaMemset(d_sum, 0, size);

   // Launch Kernel
   pi_calculation << <dimGrid, dimBlock >> > (d_sum, STEPS, base, THREADS, BLOCKS);

   // Sync
   cudaDeviceSynchronize();

   // Copy results from device to host
   cudaMemcpy(h_sum, d_sum, size, cudaMemcpyDeviceToHost);

   // Do the final reduction.
   for (threadidx = 0; threadidx < THREADS * BLOCKS; threadidx++)
      pi += h_sum[threadidx];

   // Multiply by base
   pi *= base;

   // Output Results
   printf("PI = %.10f\n", pi);

   // Cleanup
   free(h_sum);
   cudaFree(d_sum);

   return 0;
}
