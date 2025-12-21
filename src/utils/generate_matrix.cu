#include <cstdlib>

#include "generate_matrix.cuh"

void generate_matrix(float* matrix, int size) {
	for (int i = 0; i < size; i++) {
		matrix[i] = static_cast<float>(rand()) / RAND_MAX;
	}
}