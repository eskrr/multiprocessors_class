#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include "matrix.h"

#define DEBUG true
#define CUDA true
#define NUM_TESTS 5

clock_t start, end;

bool runSerial(const MATRIX mA, const MATRIX mB, MATRIX* mC, double* time) {

	clock_t start, end;
	int i = 0;
	for (; i < 0; i++) {
		start = clock();
		multiplyMatrix(
		/* startPos */ 	0,
		/* endPos */ 	mC->rows * mC->cols,
		/* matrix A */ 	mA,
		/* matrix B */ 	mB,
		/* matrix C */ 	mC);
    	end = clock();

    	*(time + i) = ((double) (end - start)) / CLOCKS_PER_SEC;

    	memset(mC->vals, 0, (mC->rows * mC->cols)*sizeof(double));
	}

	return true;
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

	if (!runSerial(*mA, *mB, mC, serialTimes)) {
		printf("Error running serial tests.\n");
		return -1;
	}

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

	freeMatrix(mA, false);
	freeMatrix(mB, false);

	int i;
	for (i = 0; i < NUM_TESTS; i++)
		printf("%5lf \n", *(serialTimes + i));

	return 0;
}