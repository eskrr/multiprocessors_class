#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include "omp.h"
#include "matrix.h"

#define DEBUG false
#define CUDA true
#define NUM_TESTS 5
#define MAX_THREADS 2014
#define MAX_BLOCKS 1024

clock_t start, end;

void runSerial(const MATRIX mA, const MATRIX mB, MATRIX* mC, double* times) {
	clock_t start, end;
	int i = 0;

	double totalTime;
	for (; i < NUM_TESTS; i++) {
		start = clock();
		multiplyMatrix(
		/* startPos */ 	0,
		/* endPos */ 	mC->rows * mC->cols,
		/* matrix A */ 	mA,
		/* matrix B */ 	mB,
		/* matrix C */ 	mC);
    	end = clock();

    	totalTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    	*(times + i) = totalTime;
    	memset(mC->vals, 0, (mC->rows * mC->cols)*sizeof(double));
	}
}

void runOmp(MATRIX* mA, MATRIX* mB, MATRIX* mC, double* times) {
	clock_t start, end;
	int i = 0;

	int totalWork = mC->rows * mC->cols;

	int totalThreads = omp_get_max_threads();
	if (totalThreads > totalWork) {
		totalThreads = totalWork;
	}

	int workPerThread = totalWork / totalThreads;

	double totalTime;
	for (; i < NUM_TESTS; i++) {
		start = clock();
		#pragma omp parallel num_threads(totalThreads) shared(workPerThread, mA, mB, mC)
		{
			int startPos = omp_get_thread_num() * workPerThread;
			int endPos = startPos + workPerThread;

			multiplyMatrix(
				/* startPos */	startPos,
				/* endPos */	endPos,
				/* matrix A */	*mA,
				/* matrix B */	*mB,
				/* matrix C */	mC);
		}
    	end = clock();

    	totalTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    	*(times + i) = totalTime;
    	memset(mC->vals, 0, (mC->rows * mC->cols)*sizeof(double));
	}
}

__global__ void calculateMatrixCuda(int *workPerThread, MATRIX* mA, MATRIX* mB, MATRIX* mC) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate index for each thread
	int pos = idx * *workPerThread;
	int endPos = pos + *workPerThread;

	int n = mA->cols;
	for (; pos < mC->rows * mC->cols && pos < endPos; pos++) {
		int row = pos / mC->rows;
		int col = pos % mC->rows;

		double sum = 0.0;
		int i = 0;
		for (; i < n; i++) {
			sum += (*(mA->vals + mA->cols * row + i) * *(mB->vals + mB->cols * col + i));
		}

		*(mC->vals + pos) = sum;
	}
}

void runCuda(MATRIX* mA, MATRIX* mB, MATRIX* mC, double* times) {
	clock_t start, end;
	int i = 0;

	int *workPerThread;
	cudaMallocManaged(&workPerThread, sizeof(int));

	int totalBlocks = mC->rows < MAX_BLOCKS ?  mC->rows : MAX_BLOCKS;
	int totalRows = mC->cols < MAX_THREADS ?  mC->cols : MAX_THREADS;

	int totalWork = mC->rows * mC->cols;
	*workPerThread = totalWork / (totalBlocks * totalRows);

	double totalTime;
	for (; i < NUM_TESTS; i++) {
		start = clock();

		calculateMatrixCuda <<<totalBlocks, totalRows>>> (workPerThread, mA, mB, mC);
		cudaDeviceSynchronize();

    	end = clock();

    	totalTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    	*(times + i) = totalTime;
    	printMatrix(*mC, 'C');
    	memset(mC->vals, 0, (mC->rows * mC->cols)*sizeof(double));
	}
}

int main(int argc, char *argv[]) {
	if (!verifyArgs(argc))
		return false;

	MATRIX *mA, *mB;
	if (!initializeInputMatrixes(argc, argv, &mA, &mB, DEBUG, CUDA))
		return -1;

	MATRIX* mC;
	if  ((mC = initializeOutputMatrix(*mA, *mB, CUDA)) == NULL) {
		printf("Error allocating output matrix C.\n");
		return false;
	}

	MATRIX *mBT = transposeMatrix(*mB, CUDA);

	freeMatrix(mB, CUDA);

	mB = mBT;

	if (DEBUG) {
		printf("Matrix B Transposed: \n");
		printMatrix(*mBT, 'T');
	}

	double *serialTimes = (double *)malloc(NUM_TESTS * sizeof(double));
	runSerial(*mA, *mB, mC, serialTimes);

	double *ompTimes = (double *)malloc(NUM_TESTS * sizeof(double));
	runOmp(mA, mB, mC, ompTimes);

	double *cudaTimes = (double *)malloc(NUM_TESTS * sizeof(double));
	runCuda(mA, mB, mC, cudaTimes);

	freeMatrix(mA, CUDA);
	freeMatrix(mB, CUDA);
	freeMatrix(mC, CUDA);

	printf("%20s %20s %20s\n", "SERIAL", "OMP", "CUDA");

	int i;
	for (i = 0; i < NUM_TESTS; i++)
		printf("%20lf %20lf %20lf\n", *(serialTimes + i), *(ompTimes + i), *(cudaTimes + i));

	printf("\n");

	return 0;
}