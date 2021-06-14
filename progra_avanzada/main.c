#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define MIN_SEQS 100
#define NUM_THREADS 8

char* reference;
char** sequences;

typedef struct mapSequenceToReferenceArgs {
	int threadId;
	char* sequence;
	int sequenceNum;
} mapSequenceToReferenceArgs;

typedef struct mapSequencesToReferenceArgs {
	int threadId;
	char** sequences;
	int totalSeqs;
} mapSequencesToReferenceArgs;

void uploadReference(char* fileName) {
	FILE* fp = fopen(fileName, "r");

	if (!fp)
		return;

	if (reference)
		free(reference);

	fseek(fp, 0L, SEEK_END);

	reference = malloc(ftell(fp));

	if (!reference)
		return;

	fseek(fp, 0, SEEK_SET);
	// Read by chunks?
	fscanf(fp, "%s", reference);
	fclose(fp);
}

void readSequences(char* fileName, char*** sequences, int* totalSeqs) {
	FILE* fp = fopen(fileName, "r");

	if (!fp)
		return;

	*sequences = malloc(sizeof(char*) * 1000);

	if (!*sequences)
		return;

	fpos_t currentPos, prevPos;
	fgetpos(fp, &prevPos);
	*totalSeqs = 0;


	while (!feof(fp)) {
		char* aux = malloc(sizeof(char));
		fscanf(fp, "%[^\n]c", aux);

		fgetpos(fp, &currentPos);
		*(*sequences + *totalSeqs) = malloc(sizeof(char) * (currentPos - prevPos));

		fsetpos(fp, &prevPos);
		fscanf(fp, "%s\n", *(*sequences + *totalSeqs));

		(*totalSeqs)++;
		fgetpos(fp, &prevPos);

		free(aux);

		// if (*totalSeqs > 1)
		// 	break;
	}
}

void *mapSequenceToReferenceImpl(void *vargp) {
	mapSequenceToReferenceArgs* args;
	args = (mapSequenceToReferenceArgs*)vargp;

	int workPerThread = strlen(reference) / NUM_THREADS;
	int current = args->threadId * workPerThread;
	int end = current + workPerThread;

	for (; current < end; current++) {
		// if (current + strlen(args->sequence) > strlen(reference))
		// 	return NULL;
		if (memcmp(reference + current, args->sequence, strlen(args->sequence)) == 0) {
			printf("%d a partir del caracter %d\n", args->sequenceNum, current);
		}
	}	

	return NULL;
}

void mapSequenceToReference(char* sequence, int sequenceNum) {
	pthread_t threads[NUM_THREADS];
	mapSequenceToReferenceArgs args[NUM_THREADS];
	int threadId;

	for (threadId = 0; threadId < NUM_THREADS; threadId++) {
    	args[threadId].threadId = threadId;
    	args[threadId].sequence = sequence;
    	args[threadId].sequenceNum = sequenceNum;
    	if (pthread_create(&threads[threadId], NULL, mapSequenceToReferenceImpl, (void*)&args[threadId]) != 0) {
    		printf("Error!\n");
    		return;
    	}

	}

	for (threadId = 0; threadId < NUM_THREADS; threadId++) {
    	if (pthread_join(threads[threadId], NULL) != 0)
    		return;
  	}
}

void *mapSequencesToReferenceImpl(void *varpg) {
	mapSequencesToReferenceArgs* args;
	args = (mapSequencesToReferenceArgs*)varpg;

	int workPerThread = args->totalSeqs / NUM_THREADS;
	int current = args->threadId * workPerThread;
	int end = current + workPerThread;

	// printf("Thread: %d (%d -> %d)\n", args->threadId, current, end);

	for (; current < end; current++) {
		// printf("Current: %d\n", current);
		mapSequenceToReference(*(args->sequences + current), current);
	}

	return NULL;
}

void mapSequencesToReference(char** sequences, const int totalSeqs) {
	// static int sequenceNum = 0;

	pthread_t threads[NUM_THREADS];
	mapSequencesToReferenceArgs args[NUM_THREADS];
	int threadId;

	for (threadId = 0; threadId < NUM_THREADS; threadId++) {
    	args[threadId].threadId = threadId;
    	args[threadId].sequences = sequences;
    	args[threadId].totalSeqs = totalSeqs;
    	if (pthread_create(&threads[threadId], NULL, mapSequencesToReferenceImpl, (void*)&args[threadId]) != 0) {
    		printf("Error!\n");
    		return;
    	}

	}

	for (threadId = 0; threadId < NUM_THREADS; threadId++) {
    	if (pthread_join(threads[threadId], NULL) != 0)
    		return;
  	}

  	// sequenceNum++;


	// FILE* fp = fopen(sequencesFileName, "r");

	// if (!fp)
	// 	return;

	// char *sequence;

	// fpos_t currentPos, prevPos;
	// fgetpos(fp, &prevPos);
	// int totalSeqs = 0;

	// while (!feof(fp)) {
	// 	char* aux = malloc(sizeof(char));
	// 	fscanf(fp, "%[^\n]c", aux);

	// 	fgetpos(fp, &currentPos);
	// 	sequence = malloc(sizeof(char) * (currentPos - prevPos));

	// 	fsetpos(fp, &prevPos);
	// 	fscanf(fp, "%s\n", sequence);

	// 	totalSeqs++;
	// 	fgetpos(fp, &prevPos);

	// 	mapSequenceToReference(sequence);

	// 	free(aux);
	// 	free(sequence);

	// 	if (totalSeqs > 2)
	// 		break;
	// }
}

// void mapSequencesToReference(char* sequencesFileName) {
// 	FILE* fp = fopen(sequencesFileName, "r");

// 	if (!fp)
// 		return;

// 	while (!feof(fp)) {

// 	}
// 	// int i = 0, j = 0;
// 	// for (; i < totalSeqs; i++) {
// 	// 	printf("ref %d: %s\n", i, refs[i]);
// 	// }

// 	// for (i = 0; i < strlen(reference) / 16; i++) {
// 		// printf("%d\n", i);
// 		// for (j = 0; j < totalSeqs; j++) {
// 		// 	if (i + strlen(sequences[j]) > strlen(reference))
// 		// 		continue;
// 		// 	if (memcmp(reference + i, sequences[j], strlen(sequences[j])) == 0) {
// 		// 		printf("%s a partir del caracter %d\n", sequences[j], i);
// 		// 	}
// 		// }
// 	// }

// 	// pthread_t* threads;

// 	// for (coreNum = 0; coreNum < virtualCores; coreNum++) {
//  //      coreNums[coreNum] = coreNum;
//  //      pthread_create(&tid[coreNum], NULL, calculatePi, (void *)&coreNums[coreNum]);
//  //  	}

// }

int main() {
	uploadReference("reference.seq");

	char** sequences;
	int totalSeqs;


	readSequences("sequences.seq", &sequences, &totalSeqs);

	mapSequencesToReference(sequences, totalSeqs);

	free(reference);

	int i = 0;
	for (;i < totalSeqs; i++) {
		// printf("%s\n", sequences[i]);
		free(sequences[i]);
	}

	free(sequences);


	return 1;
}

