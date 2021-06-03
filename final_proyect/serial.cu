#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include "matrix.h"

#define DEBUG true

clock_t start, end;


int main(int argc, char *argv[]) {
	if (!verifyArgs(argc))
		return false;

	MATRIX *mA, *mB;
	if (!initializeInputMatrixes(argc, argv, &mA, &mB, DEBUG, false))
		return -1;

	MATRIX *mBT = transposeMatrix(*mB, false);

	if (DEBUG) {
		printf("Matrix B Transposed: \n");
		printMatrix(*mBT, 'T');
	}

	MATRIX* mC;

	if  ((mC = initializeOutputMatrix(*mA, *mB, false)) == NULL) {
		printf("Error allocating output matrix C.\n");
		return -1;
	}
   
   	start = clock();
	multiplyMatrix(
		/* startPos */ 	0,
		/* endPos */ 	mC->rows * mC->cols,
		/* matrix A */ 	*mA,
		/* matrix B */ 	*mB,
		/* matrix C */ 	mC);
    end = clock();

	if (DEBUG)
		printMatrix(*mC, 'C');

 	double totalTime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total time taken by CPU: %lf\n", end - start); 

	freeMatrix(mA, false);
	freeMatrix(mB, false);
	freeMatrix(mC, false);

	return 0;
}