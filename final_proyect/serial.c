#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "matrix.h"

#define DEBUG true
// char* matrixAFileName;
// char* matrixBFileName;

int main(int argc, char *argv[]) {
	if (!verifyArgs(argc))
		return -1;

	MATRIX* mA;

	if  ((mA = initializeInputMatrix(
			argv, MatrixAFileNameArgPos,
			MatrixARowsArgPos, MatrixAColsArgPos)) == NULL) {
		printf("Error allocating matrix A.\n");
		return -1;
	}

	if (DEBUG)
		printf("Reading matrix from: %s (rows: %d, cols: %d)\n",
			mA->fileName, mA->rows, mA->cols);

	if (!readMatrix(mA)) {
		printf("Error reading matrix A from: %s\n", mA->fileName);
	} else if (DEBUG) {
		printMatrix(*mA, 'A');
	}

	MATRIX* mB;

	if  ((mB = initializeInputMatrix(
			argv, MatrixBFileNameArgPos,
			MatrixBRowsArgPos, MatrixBColsArgPos)) == NULL) {
		printf("Error allocating matrix B.\n");
		return -1;
	}

	if (DEBUG)
		printf("Reading matrix from: %s (rows: %d, cols: %d)\n",
			mB->fileName, mB->rows, mB->cols);

	if (!readMatrix(mB)) {
		printf("Error reading matrix B from: %s\n", mB->fileName);
	} else if (DEBUG) {
		printMatrix(*mB, 'B');
	}

	MATRIX* mC;

	if  ((mC = initializeOutputMatrix(*mA, *mB)) == NULL) {
		printf("Error allocating output matrix C.\n");
		return -1;
	}

	multiplyMatrix(*mA, *mB, mC);

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