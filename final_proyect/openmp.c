#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "matrix.h"


int main() {
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

	#pragma omp parallel num_threads(totalThreads) shared(totalWork, workPerThread)
	{
		printf("Thread Id: %d\n", omp_get_thread_num());
		printf("Total work: %d\n", totalWork);
		printf("Work Per work: %d\n", workPerThread);

		int startPos = omp_get_thread_num() * workPerThread;
		int endPos = startPos + workPerThread;

		printf("Start pos: ", startPos);
		printf("End pos: ", endPos);


	// 	#pragma omp master
	// 	{
	// 		printf("Hello from master = %d \n", omp_get_thread_num());
	// 	}
	// 	printf("Hello from thread = %d \n", omp_get_thread_num());
	}
	// printf("Max number of threads: %d", omp_get_max_threads());
	// printf("Current number of threads: %d", OMP_NUM_THREADS);

	freeMatrix(mA);
	freeMatrix(mB);
	freeMatrix(mC);

	return 1;
}