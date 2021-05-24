#include <stdio.h>
#include <time.h>

#define virtualCores 1000
#define intervals 1000000000
#define intervalsPerCore ((intervals)/(virtualCores))
#define intervalBase ((1.0)/(intervals))

__global__ void calculatePi(float* acums) {
   int coreNum = threadIdx.x;

   int currentInterval = coreNum * intervalsPerCore;
   int lastInterval = currentInterval + intervalsPerCore;
   double x = currentInterval * intervalBase;
   double fdx;

   for (; currentInterval < lastInterval; currentInterval++) {
      fdx = 4 / (1 + x * x);
      acums[coreNum] = acums[coreNum] + (fdx * intervalBase);
      x = x + intervalBase;
   }
}

void main() {
   // Initialize host variables
   float *h_acums;
   int size = sizeof(float) * virtualCores;

   h_acums = (float *)malloc(size);

   // Initialize device variables
   float *d_acums;
   cudaMalloc((void**)&d_acums, size);

   clock_t start, end;

   start = clock();
   calculatePi <<<1, virtualCores >>> (d_acums);

   // Wait for device
   cudaDeviceSynchronize();
   cudaMemcpy(h_acums, d_acums, size, cudaMemcpyDeviceToHost);

   int coreNum;
   double acum = 0.0;
   for (coreNum = 0; coreNum < virtualCores; coreNum++) {
      acum += d_acums[coreNum];
   }

   end = clock();

   printf("Result = %20.18lf (%ld)\n", acum, end - start);
}