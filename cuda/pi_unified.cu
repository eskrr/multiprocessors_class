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
   double *sum;  // Pointer to host & device arrays
   double base = 1.0 / STEPS;  // base size
   size_t size = BLOCKS * THREADS * sizeof(double);  //Array memory size

   // Memory allocation
   cudaMallocManaged(&sum, size);  // Allocate array on device

   // Initialize array in device to 0
   cudaMemset(sum, 0, size);

   clock_t start, end;

   start = clock();

   // Launch Kernel
   pi_calculation << <dimGrid, dimBlock >> > (sum, STEPS, base, THREADS, BLOCKS);

   // Sync
   cudaDeviceSynchronize();

   // Do the final reduction.
   for (threadidx = 0; threadidx < THREADS * BLOCKS; threadidx++)
      pi += sum[threadidx];

   // Multiply by base
   pi *= base;

   end = clock();

   // Output Results
   printf("Result = %20.18lf (%ld)\n", pi, end - start);

   // Cleanup
   cudaFree(sum);

   return 0;
}
