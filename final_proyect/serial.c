#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "matrix.h"

#define DEBUG true

int main(int argc, char *argv[]) {
	if (!verifyArgs(argc))
		return false;


	MATRIX *mA, *mB;
	if (!initializeInputMatrixes(argc, argv, &mA, &mB, DEBUG))
		return -1;

	MATRIX* mC;

	if  ((mC = initializeOutputMatrix(*mA, *mB)) == NULL) {
		printf("Error allocating output matrix C.\n");
		return -1;
	}

	multiplyMatrix(
		/* startPos */ 0,
		/* endPos */ mC->rows * mC->cols,
		/* matrix A */ *mA,
		/* matrix B */ *mB,
		/* matrix C */ mC);

	printMatrix(*mC, 'C');

	// if (DEBUG)
	// 	printf("Reading matrix from: %s (rows: %d, cols: %d)\n",
	// 		mB->fileName, mB->rows, mB->cols);

	// if (!readMatrix(mB)) {
	// 	printf("Error reading matrix B from: %s\n", mB->fileName);
	// }

	// if (argc < 3) {
	// 	printf("Please provide filenames for matrix A and B and their respective sizes in the following format: \n");
	// 	printf("./binaryName matrixAFileName numRows numCols matrixBFileName numRows numCols");
	// 	return -1;
	// }
	// matrixAFileName = argv[1];
	// matrixBFileName = argv[2];

	// MATRIX* mA = readMatrix('A', matrixAFileName);
	// if (!mA) {
	// 	printf("Error reading matrix A\n");
	// 	return -1;
	// }
	// if (DEBUG)
	// 	printMatrix(*mA);

	// MATRIX* mB = readMatrix('A', matrixBFileName);
	// if (!mB) {
	// 	printf("Error reading matrix B\n");
	// 	return -1;
	// }
	// if (DEBUG)
	// 	printMatrix(*mB);
	freeMatrix(mA);
	freeMatrix(mB);
	freeMatrix(mC);

	return 0;
}