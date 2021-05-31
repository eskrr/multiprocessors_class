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

void freeMatrix(MATRIX* matrix) {
	if (matrix == NULL)
		return;
	if (matrix->fileName != NULL)
		free(matrix->fileName);
	if (matrix->vals != NULL)
		free(matrix->vals);

	free(matrix);
}

bool verifyArgs(int totalArguments) {
	if (totalArguments < 7) {
		printf("Please provide filenames for matrix A and B and their respective sizes in the following format: \n");
		printf("./binaryName matrixAFileName numRows numCols matrixBFileName numRows numCols\n");
		return false;
	}
	return true;
}

MATRIX* initializeOutputMatrix(
	const MATRIX mA,
	const MATRIX mB) {
	MATRIX* output = malloc(sizeof(MATRIX));
	if (mA.cols != mB.rows) {
		printf("Number of cols of matrix A must be equal to number of rows in matrix B.\n");
		return NULL;
	}

	output->fileName = NULL;
	output->rows = mA.rows;
	output->cols = mB.cols;

	output->vals = calloc(output->rows * output->cols, sizeof(double));

	return output;
}

MATRIX* initializeInputMatrix(
	char *arguments[], const int fileNameArgPos,
	const int rowsArgPos, const int colsArgPos) {
	MATRIX* m = malloc(sizeof(MATRIX));
	if (m == NULL)
		return NULL;

	m->fileName = strdup(arguments[fileNameArgPos]);
	m->rows = atoi(arguments[rowsArgPos]);
	m->cols = atoi(arguments[colsArgPos]);

	m->vals = calloc(m->rows * m->cols, sizeof(double));
	if (m->vals == NULL) {
		freeMatrix(m);
		return NULL;
	}

	return m;
}

bool readMatrix(MATRIX* matrix) {
	FILE* fp = fopen(matrix->fileName, "r");
    if(!fp)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        return false;

	int pos = 0;
	for (; pos < matrix->rows * matrix->cols; pos++)
		if (fscanf(fp, "%lf", matrix->vals + pos) != 1)
			return false;

	fclose(fp);

	return true;
}

void printMatrix(const MATRIX matrix, const char letter) {
	int pos;
	printf("Printing matrix: %c\n", letter);
	for (pos = 0; pos < matrix.rows * matrix.cols; pos++) {
		if (pos % matrix.cols == 0)
			printf("\n");
		printf("%20lf ", *(matrix.vals + pos));
	}
	printf("\n");
}

double* matrixValue(const MATRIX matrix, const int row, const int col) {
	return matrix.vals + matrix.cols * row + col;
}

void multiplyMatrix(const MATRIX mA, const MATRIX mB, MATRIX* mC) {
	int n = mA.cols;
	int row, col;
	int cont = 0;
	for (row = 0; row < mC->rows; row++) {
		for (col = 0; col < mC->cols; col++) {
			// int pos = 
			// printf("%d", mC->rows * row + col);
			// int i = 0;
			// for ()
			int i = 0;
			double sum = 0.0;
			for (; i < n; i++) {
				double valA, valB;
				valA = *matrixValue(mA, row, i);
				valB = *matrixValue(mB, i, col);

				sum += valA * valB;

			}
			// printf("\n");
			// printf("%d, %d = %lf\n", row, col, sum);

			double* newVal = matrixValue(*mC, row, col);
			*newVal = sum;
		}
	}
}