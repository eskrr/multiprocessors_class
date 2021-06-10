#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"

#define MIN_SEQS 100
#define numRefs 5

char seq[] = "ACAAGATGCCATTGTCCCCCGGCCTCCTGCTGCTGCTGCTCTCCGGGGCCACGGCCACCGCTGCCCTGCCCCTGGAGGGTGGCCCCACCGGCCGAGACAGCGAGCATATGCAGGAAGCGGCAGGAATAAGGAAAAGCAGCCTCCTGACTTTCCTCGCTTGGTGGTTTGAGTGGACCTCCCAGGCCAGTGCCGGGCCCCTCATAGGAGAGGAAGCTCGGGAGGTGGCCAGGCGGCAGGAAGGCGCACCCCCCCAGCAATCCGCGCGCCGGGACAGAATGCCCTGCAGGAACTTCTTCTGGAAGACCTTCTCCTCCTGCAAATAAAACCTCACCCATGAATGCTCACGCAAGTTTAATTACAGACCTGAA";
char *refs[] = {
	"GCCTCCTGCTGCTGCTGCTCTCC", // 21
	"GGACCTCCCAGGCCAGTGCCGGG", // 171
	"AAGACCTTCTCCTCCTGCAAATA", // 299
	"TTCTTCTGGAAGACCTTCTCCTC", // 290 
	"CCAGGCGGCAGGAAGGCGCACCCCCCCAGCAATCCGTGCGCCGG",
};


char* reference;

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

	// Start array size at minimum size.
	*sequences = malloc(sizeof(char *) * MIN_SEQS);

	// todo: throw error handling
	if (!*sequences )
		return;


	fpos_t currentPos, prevPos;
	fgetpos(fp, &currentPos);
	*totalSeqs = 0;
	int reallocCount = 0;

	while (!feof(fp)) {
		char* aux = malloc(sizeof(char));
		fscanf(fp, "%[^\n]c", aux);

		fgetpos(fp, &currentPos);
		// printf("%lld - %lld\n: %d\n", prevPos, currentPos, totalSeqs);
		*(*sequences  + *totalSeqs) = malloc(sizeof(char) * ((int)currentPos - (int)prevPos));

		fsetpos(fp, &prevPos);
		fscanf(fp, "%s\n", *(*sequences  + *totalSeqs));

		(*totalSeqs)++;
		fgetpos(fp, &prevPos);
		free(aux);

		if (*totalSeqs >= (reallocCount * MIN_SEQS + MIN_SEQS)) {
			*sequences  = realloc(*sequences , (*totalSeqs + MIN_SEQS) * sizeof(char *));

			// throw error handling
			if (!*sequences)
				return;

			reallocCount++;
		}
	}
	// free(sequences[0]);
}

void mapSequencesToReference(char** sequences, int totalSeqs) {
	int i = 0, j = 0;
	// for (; i < totalSeqs; i++) {
	// 	printf("ref %d: %s\n", i, refs[i]);
	// }

	for (i = 0; i < strlen(reference); i++) {
		printf("%d\n", i);
		// for (j = 0; j < totalSeqs; j++) {
		// 	if (i + strlen(sequences[j]) > strlen(reference))
		// 		continue;
		// 	if (memcmp(reference + i, sequences[j], strlen(sequences[j])) == 0) {
		// 		printf("%s a partir del caracter %d\n", sequences[j], i);
		// 	}
		// }
	}
}

int main() {
	printf("%d\n", omp_get_max_threads());

	// printf("Sequence: %s\n", seq);


	// uploadReference("reference.seq");
	// char** seqs;
	// int totalSeqs;

	// readSequences("sequences.seq", &seqs, &totalSeqs);

	// mapSequencesToReference(seqs, totalSeqs);

	// int i;
	// // printf("Total length: %d\n", totalSeqs);
	// for (i = 0; i < totalSeqs; i++) {
	// 	// if (i < 5)
	// 	// 	printf("%s\n", seqs[i]);
	// 	free(seqs[i]);
	// }

	// free(seqs);

	// free(reference);

	return 1;
}

