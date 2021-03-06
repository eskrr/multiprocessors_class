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
#define MAX_THREADS 1024
#define MAX_BLOCKS 1024
#define OUTPUT_FILE "matrixC.txt"

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

    	totalTime = ((double) (end - start)) / (CLOCKS_PER_SEC / 1000);
    	*(times + i) = totalTime;
    	// memset(mC->vals, 0, (mC->rows * mC->cols)*sizeof(double));
	}
}

void runOmp(MATRIX* mA, MATRIX* mB, MATRIX* mC, double* times, const MATRIX mCSerial) {
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
		int workLeft = totalWork % totalThreads;
		if (workLeft) {
			#pragma omp parallel num_threads(workLeft) shared(totalThreads, workPerThread, mA, mB, mC)
			{
				int startPos = totalThreads * workPerThread + omp_get_thread_num();
				int endPos = startPos + 1;

				multiplyMatrix(
					/* startPos */	startPos,
					/* endPos */	endPos,
					/* matrix A */	*mA,
					/* matrix B */	*mB,
					/* matrix C */	mC);
			}
		}

    	end = clock();

    	totalTime = ((double) (end - start)) / ( CLOCKS_PER_SEC / 1000);
    	*(times + i) = totalTime;

    	if (compareMatrixes(*mC, mCSerial)) {
    		printf("OMP: Matrixes are equal.\n");
    	} else {
    		printf("OMP: Error matrixes are not equal.\n");
    	}

    	memset(mC->vals, 0, (mC->rows * mC->cols)*sizeof(double));
	}
}

__global__ void calculateMatrixCuda(int *workPerThread, MATRIX* mA, MATRIX* mB, MATRIX* mC, int offset) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;  // Calculate index for each thread
	int pos = idx * *workPerThread;
	int endPos = pos + *workPerThread;

	int n = mA->cols;
	for (; pos < mC->rows * mC->cols && pos < endPos; pos++) {
		int row = pos / mC->cols;
		int col = pos % mC->cols;

		double sum = 0.0;
		int i = 0;
		for (; i < n; i++) {
			sum += (*(mA->vals + mA->cols * row + i) * *(mB->vals + mB->cols * col + i));
		}

		*(mC->vals + pos) = sum;
	}
}

void runCuda(MATRIX* mA, MATRIX* mB, MATRIX* mC, double* times, const MATRIX mCSerial) {
	clock_t start, end;
	int i = 0;

	int *workPerThread;
	cudaMallocManaged(&workPerThread, sizeof(int));

	int totalBlocks = mC->rows < MAX_BLOCKS ?  mC->rows : MAX_BLOCKS;
	int totalRows = mC->cols < MAX_THREADS ?  mC->cols : MAX_THREADS;

	int totalWork = mC->rows * mC->cols;
	*workPerThread = totalWork / (totalBlocks * totalRows);

	int workLeft = totalWork % (totalBlocks * totalRows);

	double totalTime;
	for (; i < NUM_TESTS; i++) {
		start = clock();

		calculateMatrixCuda <<<totalBlocks, totalRows>>> (workPerThread, mA, mB, mC, 0);
		if (workLeft)
			calculateMatrixCuda <<<1, workLeft>>> (workPerThread, mA, mB, mC, totalRows * totalBlocks);
		cudaDeviceSynchronize();

    	end = clock();

    	totalTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    	*(times + i) = totalTime;

    	if (compareMatrixes(*mC, mCSerial)) {
    		printf("CUDA: Matrixes are equal.\n");
    	} else {
    		printf("CUDA: Error matrixes are not equal.\n");
    	}

    	memset(mC->vals, 0, (mC->rows * mC->cols)*sizeof(double));
	}

	cudaFree(workPerThread);
}

void saveMatrix(const MATRIX mC) {
	FILE* fp = fopen(OUTPUT_FILE, "w");

	int i = 0;
	for (; i < mC.rows * mC.cols; i++) {
		fprintf(fp, "%0.10lf\n", *(mC.vals + i));
	}

	fclose(fp);
}

void printComparison(double *serialTimes, double *ompTimes, double *cudaTimes) {
	printf("%20s %20s %20s %20s\n", "RUN", "SERIAL", "OMP", "CUDA");

	int i;
	double serialSum = 0.0, ompSum = 0.0, cudaSum = 0.0;

	printf("\n");
	for (i = 0; i < NUM_TESTS; i++) {
		serialSum += *(serialTimes + i);
		ompSum += *(ompTimes + i);
		cudaSum += *(cudaTimes + i);
		printf("%20d %20.10lf %20.10lf %20.10lf\n", i + 1, *(serialTimes + i), *(ompTimes + i), *(cudaTimes + i));
	}

	double serialAvg = serialSum / NUM_TESTS;
	double ompAvg = ompSum / NUM_TESTS;
	double cudaAvg = cudaSum / NUM_TESTS;
	printf("%20s %20.10lf %20.10lf %20.10lf", "Promedio", serialAvg, ompAvg, cudaAvg);
	printf("\n");
	printf("%20s %20.10lf %20.10lf %20.10lf", "% vs Serial", serialAvg / serialAvg, ompAvg / serialAvg, cudaAvg / serialAvg);
	printf("\n");

	if (serialAvg < ompAvg && serialAvg < cudaAvg) {
		printf("Serial was fastest.\n");
	}

	if (ompAvg < serialAvg && ompAvg < cudaAvg) {
		printf("OMP was fastest.\n");
	}

	if (cudaAvg < serialAvg && cudaAvg < ompAvg) {
		printf("CUDA was fastest.\n");
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

	MATRIX* mCParallel;
	if  ((mCParallel = initializeOutputMatrix(*mA, *mB, CUDA)) == NULL) {
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
	runOmp(mA, mB, mCParallel, ompTimes, *mC);

	double *cudaTimes = (double *)malloc(NUM_TESTS * sizeof(double));
	runCuda(mA, mB, mCParallel, cudaTimes, *mC);

	saveMatrix(*mC);

	freeMatrix(mA, CUDA);
	freeMatrix(mB, CUDA);
	freeMatrix(mC, CUDA);
	freeMatrix(mCParallel, CUDA);

	printComparison(serialTimes, ompTimes, cudaTimes);

	free(serialTimes);
	free(ompTimes);
	free(cudaTimes);

	return 0;
}