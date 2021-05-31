#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#define MatrixAFileNameArgPos 1
#define MatrixARowsArgPos 2
#define MatrixAColsArgPos 3
#define MatrixBFileNameArgPos 4
#define MatrixBRowsArgPos 5
#define MatrixBColsArgPos 6

typedef struct MATRIX {
	char* fileName;
	double* vals;
	int rows;
	int cols;
} MATRIX;

bool verifyArgs(int totalArguments) {
	if (totalArguments < 7) {
		printf("Please provide filenames for matrix A and B and their respective sizes in the following format: \n");
		printf("./binaryName matrixAFileName numRows numCols matrixBFileName numRows numCols\n");
		return false;
	}
	return true;
}

MATRIX* initializeInputMatrix(
	char *arguments[], const int fileNameArgPos,
	const int rowsArgPos, const int colsArgPos) {
	MATRIX* m = malloc(sizeof(MATRIX));

	m->fileName = strdup(arguments[fileNameArgPos]);
	m->rows = atoi(arguments[rowsArgPos]);
	m->cols = atoi(arguments[colsArgPos]);

	return m;
}

bool readMatrix(MATRIX* matrix) {
	FILE* fp = fopen(matrix->fileName, "r");
    if(!fp)
        return false;

	matrix->vals = calloc(matrix->rows * matrix->cols, sizeof(char));

	int pos = 0;
	for (; pos < matrix->rows * matrix->cols; pos++) {
		double val;
		if (fscanf(fp, "%lf", &val) != 1)
			return false;
		printf("%20lf ", val);
		*(matrix->vals + pos) = val;
	}

	fclose(fp);

	return true;
}

void printMatrix(const MATRIX matrix) {
	int pos;
	printf("Printing matrix from %s: ..\n", matrix.fileName);
	for (pos = 0; pos < matrix.rows * matrix.cols - 7; pos++) {
		if (pos % matrix.cols == 0)
			printf("\n");
		printf("%20lf ", *(matrix.vals + pos));
	}
	printf("\n");
}

void freeMatrix(MATRIX* matrix) {
	free(matrix->fileName);
	free(matrix->vals);
	free(matrix);
}

// double getVal(const MATRIX matrix, const int row, const int col) {

// }