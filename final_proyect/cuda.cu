#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include "matrix.h"

#define DEBUG true
#define MAX_THREADS 2014
#define MAX_BLOCKS 1024

clock_t start, end;

__global__ void calculateMatrixCuda(int *workPerThread, MATRIX* mA, MATRIX* mB, MATRIX* mC) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate index for each thread
	int pos = idx * *workPerThread;
	int endPos = pos + *workPerThread;

	int n = mA->cols;
	for (; pos < mC->rows * mC->cols && pos < endPos; pos++) {
		int row = pos / mC->rows;
		int col = pos % mC->cols;

		double sum = 0.0;
		int i = 0;
		for (; i < n; i++) {
			// double valA, valB;

			// valA = *matrixValue(mA, row, i);
			// valB = *matrixValue(mB, i, col);

			sum += (*(mA->vals + mA->cols * row + i) * *(mA->vals + mA->cols * i + col));
		}

		*(mC->vals + pos) = sum;
	}

	// printf("(ThreadId: %d, WorkPerThread: %d)\n", idx, *workPerThread);
	// printf("(start: %d, end: %d)\n", startPos, endPos);
	// printf("mA: rows: %d, cols: %d\n", mA->rows, mA->cols);
	// printf("mB: rows: %d, cols: %d\n", mB->rows, mB->cols);
	// printf("mC: rows: %d, cols: %d\n", mC->rows, mC->cols);
}

int main(int argc, char *argv[]) {
	if (!verifyArgs(argc))
		return false;

	MATRIX *mA, *mB;
	if (!initializeInputMatrixes(argc, argv, &mA, &mB, DEBUG, true)) {
		printf("Error allocating input matrixes.\n");
		return -1;
	}

	// MATRIX* mC;

	// if  ((mC = initializeOutputMatrix(*mA, *mB, true)) == NULL) {
	// 	printf("Error allocating output matrix C.\n");
	// 	return -1;
	// }

	// int *workPerThread;
	// cudaMallocManaged(&workPerThread, sizeof(int));

	// int totalBlocks = mC->rows < MAX_BLOCKS ?  mC->rows : MAX_BLOCKS;
	// int totalRows = mC->cols < MAX_THREADS ?  mC->cols : MAX_THREADS;

	// int totalWork = mC->rows * mC->cols;
	// *workPerThread = totalWork / (totalBlocks * totalRows);

	// printf("totalBlocks: %d, totalRows: %d\n", totalBlocks, totalRows);

	// start = clock();
	// calculateMatrixCuda <<<totalBlocks, totalRows>>> (workPerThread, mA, mB, mC);
	// cudaDeviceSynchronize();
	// end = clock();

	// if (DEBUG)
	// 	printMatrix(*mC, 'C');

 // 	// double totalTime = (double)(end - start) / CLOCKS_PER_SEC;
 //    printf("Total time taken by CPU: %lf\n", end - start); 

	// printf("Verifying matrix... \n");	if (verifyMatrix(*mA, *mB, *mC)) {
	// 	printf("Matrix verified!!!\n");
	// }

	freeMatrix(mA, true);
	freeMatrix(mB, true);
	// freeMatrix(mC, true);

	return 0;
}