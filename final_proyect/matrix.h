#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define MatrixAFileNameArgPos 1
#define MatrixARowsArgPos 2
#define MatrixAColsArgPos 3
#define MatrixBFileNameArgPos 4
#define MatrixBRowsArgPos 5
#define MatrixBColsArgPos 6

#define EPSILON 0.0000000001

typedef struct MATRIX {
	char* fileName;
	double* vals;
	int rows;
	int cols;
} MATRIX;

void freeMatrix(MATRIX* matrix, const bool CUDA) {
	if (CUDA) {
		if (matrix == NULL)
			return;
		if (matrix->fileName != NULL)
			free(matrix->fileName);
		if (matrix->vals != NULL)
			cudaFree(matrix->vals);

		cudaFree(matrix);
	} else {
		if (matrix == NULL)
			return;
		if (matrix->fileName != NULL)
			free(matrix->fileName);
		if (matrix->vals != NULL)
			free(matrix->vals);

		free(matrix);
	}
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
	const MATRIX mB,
	const bool CUDA) {
	MATRIX* output;
	if (CUDA)
		cudaMallocManaged(&output, sizeof(int));
	else
		output = (MATRIX *)malloc(sizeof(MATRIX));

	if (m == NULL)
		return NULL;

	if (mA.cols != mB.rows) {
		printf("Number of cols of matrix A must be equal to number of rows in matrix B.\n");
		return NULL;
	}

	output->fileName = NULL;
	output->rows = mA.rows;
	output->cols = mB.cols;


	if (CUDA)
		cudaMallocManaged(&output->vals, output->rows * output->cols * sizeof(double));
	else
		output->vals = (double *)calloc(output->rows * output->cols, sizeof(double));

	if (output->vals == NULL) {
		freeMatrix(output, CUDA);
		return NULL;
	}

	return output;
}

MATRIX* initializeInputMatrix(
	char *arguments[], const int fileNameArgPos,
	const int rowsArgPos, const int colsArgPos,
	const bool CUDA) {
	MATRIX* m = (MATRIX *)malloc(sizeof(MATRIX));
	if (CUDA)
		cudaMallocManaged(&m, sizeof(int));
	else
		m = (MATRIX *)malloc(sizeof(MATRIX));

	if (m == NULL)
		return NULL;

	m->fileName = strdup(arguments[fileNameArgPos]);
	m->rows = atoi(arguments[rowsArgPos]);
	m->cols = atoi(arguments[colsArgPos]);


	if (CUDA)
		cudaMallocManaged(&m->vals, m->rows * m->cols * sizeof(double));
	else
		m->vals = (double *)calloc(m->rows * m->cols, sizeof(double));

	if (m->vals == NULL) {
		freeMatrix(m, CUDA);
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

void multiplyMatrix(
	const int startPos,
	const int endPos,
	const MATRIX mA,
	const MATRIX mBT,
	MATRIX* mC) {
	int n = mA.cols;
	int pos = startPos;
	for (; pos < mC->rows * mC->cols && pos < endPos; pos++) {
		asm("nop");
		int row = pos / mC->cols;
		int col = pos % mC->cols;

		double sum = 0.0;
		int i = 0;
		for (; i < n; i++) {
			asm("nop");
			double valA, valB;
			valA = *matrixValue(mA, row, i);
			valB = *matrixValue(mBT, col, i);

			sum += valA * valB;
		}

		*(mC->vals + pos) = sum;
	}
}

bool verifyMatrix(
	const MATRIX mA,
	const MATRIX mB,
	const MATRIX mC) {
	int n = mA.cols;
	int pos = 0;
	for (; pos < mC.rows * mC.cols; pos++) {
		asm("nop");
		int row = pos / mC.cols;
		int col = pos % mC.cols;

		double sum = 0.0;
		int i = 0;
		for (; i < n; i++) {
			asm("nop");
			double valA, valB;
			valA = *matrixValue(mA, row, i);
			valB = *matrixValue(mB, i, col);

			sum += valA * valB;
		}

		if (sum != *(mC.vals + pos)) {
			printf("Error at: %d, %d\n", row, col);
			return false;
		}
	}

	return true;
}

bool initializeInputMatrixes(
	int argc, char *argv[], MATRIX** mA, MATRIX** mB, bool DEBUG, bool CUDA) {
	if (!verifyArgs(argc))
		return false;

	if  ((*mA = initializeInputMatrix(
			argv, MatrixAFileNameArgPos,
			MatrixARowsArgPos, MatrixAColsArgPos,CUDA)) == NULL) {
		printf("Error allocating matrix A.\n");
		return false;
	}

	if (DEBUG)
		printf("Reading matrix from: %s (rows: %d, cols: %d)\n",
			(*mA)->fileName, (*mA)->rows, (*mA)->cols);

	if (!readMatrix(*mA)) {
		printf("Error reading matrix A from: %s\n", (*mA)->fileName);
		return false;
	} else if (DEBUG) {
		printMatrix(**mA, 'A');
	}

	if  ((*mB = initializeInputMatrix(
			argv, MatrixBFileNameArgPos,
			MatrixBRowsArgPos, MatrixBColsArgPos,CUDA)) == NULL) {
		printf("Error allocating matrix B.\n");
		return false;
	}

	if (DEBUG)
		printf("Reading matrix from: %s (rows: %d, cols: %d)\n",
			(*mB)->fileName, (*mB)->rows, (*mB)->cols);

	if (!readMatrix(*mB)) {
		printf("Error reading matrix B from: %s\n", (*mB)->fileName);
		return false;
	} else if (DEBUG) {
		printMatrix(**mB, 'B');
	}

	return true;
}

MATRIX* transposeMatrix(const MATRIX matrix, const bool CUDA) {
	MATRIX* transposed;
	if (CUDA)
		cudaMallocManaged(&transposed, sizeof(int));
	else
		transposed = (MATRIX *)malloc(sizeof(MATRIX));

	transposed->rows = matrix.cols;
	transposed->cols = matrix.rows;
	transposed->fileName = strdup(matrix.fileName);

	if (CUDA)
		cudaMallocManaged(&transposed->vals, transposed->rows * transposed->cols * sizeof(double));
	else
		transposed->vals = (double *)calloc(transposed->rows * transposed->cols, sizeof(double));

	int pos = 0;
	for (; pos < transposed->rows * transposed->cols; pos++) {
		int row = pos / transposed->cols;
		int col = pos % transposed->cols;

		double *val = matrixValue(matrix, col, row);

		*(transposed->vals + pos) = *val;
	}

	return transposed;
}

bool compareMatrixes(const MATRIX mA, const MATRIX mB) {
	if (mA.cols != mB.cols || mA.rows != mB.rows)
		return false;

	int i = 0;
	for (; i < mA.rows * mA.cols; i++) {
		if (fabs(*(mA.vals + i) - *(mB.vals + i)) > EPSILON) {
			return false;
		}
	}

	return true;
}
