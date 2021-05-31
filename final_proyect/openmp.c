#include <omp.h>
#include <stdio.h>

int main() {

	#pragma omp parallel num_threads(6)
	{
		#pragma omp master
		{
			printf("Hello from master = %d \n", omp_get_thread_num());
		}
		printf("Hello from thread = %d \n", omp_get_thread_num());
	}
	// printf("Max number of threads: %d", omp_get_max_threads());
	// printf("Current number of threads: %d", OMP_NUM_THREADS);

	return 1;
}