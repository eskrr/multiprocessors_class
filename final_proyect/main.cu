#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include "matrix.h"

#define DEBUG false
#define CUDA true
#define NUM_TESTS 5

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

void runOmp(const MATRIX mA, const MATRIX mB, MATRIX* mC, double* times) {
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
				/* matrix A */	mA,
				/* matrix B */	mB,
				/* matrix C */	mC);
		}
    	end = clock();

    	totalTime = ((double) (end - start)) / CLOCKS_PER_SEC;
    	*(times + i) = totalTime;
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
	runOmp(*mA, *mB, mC, ompTimes);

	// MATRIX* mC;

	// if  ((mC = initializeOutputMatrix(*mA, *mB, false)) == NULL) {
	// 	printf("Error allocating output matrix C.\n");
	// 	return -1;
	// }
   
 //   	start = clock();
	// multiplyMatrix(
	// 	/* startPos */ 	0,
	// 	/* endPos */ 	mC->rows * mC->cols,
	// 	/* matrix A */ 	*mA,
	// 	/* matrix B */ 	*mB,
	// 	/* matrix C */ 	mC);
 //    end = clock();

	// if (DEBUG)
	// 	printMatrix(*mC, 'C');


	// MATRIX* mCT;

	// if  ((mCT = initializeOutputMatrix(*mA, *mB, false)) == NULL) {
	// 	printf("Error allocating output matrix C.\n");
	// 	return -1;
	// }
   
 //   	start = clock();
	// multiplyMatrixTransposed(
	// 	/* startPos */ 	0,
	// 	/* endPos */ 	mCT->rows * mCT->cols,
	// 	/* matrix A */ 	*mA,
	// 	/* matrix B */ 	*mBT,
	// 	/* matrix C */ 	mC);
 //    end = clock();

	// if (DEBUG)
	// 	printMatrix(*mC, 'T');

 // 	double totalTime = (double)(end - start) / CLOCKS_PER_SEC;
 //    printf("Total time taken by CPU: %lf\n", end - start); 

	freeMatrix(mA, CUDA);
	freeMatrix(mB, CUDA);
	freeMatrix(mC, CUDA);

	int i;
	for (i = 0; i < NUM_TESTS; i++)
		printf("%15lf ", *(serialTimes + i));
	printf("\n");

	return 0;
}