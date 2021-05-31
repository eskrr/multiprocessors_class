#include <omp.h>
#include <stdio.h>

int main() {
	printf("Max number of threads: %d", omp_get_max_threads());
	printf("Current number of threads: %d", OMP_NUM_THREADS);

	return 1;
}