#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include "matrix.h"

#define DEBUG false

clock_t start, end;

int main(int argc, char *argv[]) {
	if (!verifyArgs(argc))
		return false;

	MATRIX *mA, *mB;
	if (!initializeInputMatrixes(argc, argv, &mA, &mB, DEBUG, true))
		return -1;

	MATRIX* mC;

	if  ((mC = initializeOutputMatrix(*mA, *mB, true)) == NULL) {
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

	start = clock();
	#pragma omp parallel num_threads(totalThreads) shared(workPerThread, mA, mB, mC)
	{
		int startPos = omp_get_thread_num() * workPerThread;
		int endPos = startPos + workPerThread;

		multiplyMatrix(
			/* startPos */ startPos,
			/* endPos */ endPos,
			/* matrix A */ *mA,
			/* matrix B */ *mB,
			/* matrix C */ mC);
	}
    end = clock();

	if (DEBUG)
		printMatrix(*mC, 'C');

 	double totalTime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Total time taken by CPU: %lf\n", end - start); 

	printf("Verifying matrix... \n");

	if (verifyMatrix(*mA, *mB, *mC)) {
		printf("Matrix verified!!!\n");
	}


	// #pragma omp parallel for shared(mA, mB, mC)
	// {
	// 	multiplyMatrix(
	// 		/* startPos */ 0,
	// 		 endPos  mC->rows * mC->cols,
	// 		/* matrix A */ *mA,
	// 		/* matrix B */ *mB,
	// 		/* matrix C */ mC);
	// }

	freeMatrix(mA);
	freeMatrix(mB);
	freeMatrix(mC);

	return 1;
}