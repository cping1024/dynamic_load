#include "tools.h"

typedef struct func_table {
	float (*sum)(const float*, int);
	float (*dot)(const float*, const float*, int);
} func_table_t;


static void slower1() {	
	double total = 1;
	for (int j = 0; j < 1000000; ++j) {
		total += j;
	}
}

void slower() {	
	double total = 1;
	for (int j = 0; j < 100000; ++j) {
		total += j;
	}

	slower1();
}

float sum(const float* vector, int len) {
	
	float sum = 0.0f;
	for (int i = 0; i < len; ++i) {
		sum += vector[i];
	}
	
	slower();
	return sum;
}

float dot(const float* a, const float* b, int len) {
	float sum = 0.0f;
	for (int i = 0; i < len; ++i) {
		sum += (a[i] * b[i]);
	}

	return sum;
}


void* register_func() {

	static func_table_t table {
		sum,
		dot
	};

	return &table;
}
