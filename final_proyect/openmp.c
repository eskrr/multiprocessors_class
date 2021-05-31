#include <omp.h>
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

	int totalWork = mC->rows * mC->cols;

	int totalThreads = omp_get_max_threads();
	if (totalThreads > totalWork) {
		totalThreads = totalWork;
	}

	printf("Starting %d threads ...\n", totalThreads);

	int workPerThread = totalWork / totalThreads;

	// #pragma omp parallel num_threads(totalThreads) shared(totalWork, workPerThread, mA, mB, mC)
	// {
	// 	printf("Thread Id: %d\n", omp_get_thread_num());
	// 	printf("Total Work: %d\n", totalWork);
	// 	printf("Work Per Thread: %d\n", workPerThread);

	// 	int startPos = omp_get_thread_num() * workPerThread;
	// 	int endPos = startPos + workPerThread;

	// 	printf("Start Pos: %d\n", startPos);
	// 	printf("End Pos: %d\n", endPos);


	// 	multiplyMatrix(
	// 		/* startPos */ startPos,
	// 		/* endPos */ endPos,
	// 		/* matrix A */ *mA,
	// 		/* matrix B */ *mB,
	// 		/* matrix C */ mC);
	// }

	#pragma omp parallel for shared(mA, mB, mC)
	{
		multiplyMatrix(
			/* startPos */ 0,
			/* endPos */ mC->rows * mC->cols,
			/* matrix A */ *mA,
			/* matrix B */ *mB,
			/* matrix C */ mC);
	}

	printMatrix(*mC, 'C');
	// printf("Max number of threads: %d", omp_get_max_threads());
	// printf("Current number of threads: %d", OMP_NUM_THREADS);

	freeMatrix(mA);
	freeMatrix(mB);
	freeMatrix(mC);

	return 1;
}