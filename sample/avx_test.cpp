#include <stdio.h>
#include <iostream>
#include <chrono>
#include <x86intrin.h>

using namespace std::chrono;

static float avx2_dot_product(const float *a, const float *b, int n) {
        float v[8];
        float prod = 0;
        int step = 8;   // avx2 __m256 can hold 8 floats
        int epoch = n / step;
        for (int i = 0; i < epoch; i++) {
                __m256 vec1 = _mm256_loadu_ps(a + i * step); // from memory to ymm1
                __m256 vec2 = _mm256_loadu_ps(b + i * step); // from memory to ymm2
                __m256 prod_sum = _mm256_dp_ps(vec1, vec2, 0xFF); // dot product
                _mm256_storeu_ps(v, prod_sum); // form ymm3 to memory
                prod += (v[0] + v[4]); // get the result
        }

        return prod;
}

static float vector_dot(float* A, float* B, int len) {
	
	float sum = 0;
	for (int i = 0; i < len; ++i) {
		sum += (A[i] * B[i]);
	}
	
	printf("sum [%f].\n", sum);
	return sum;
}

void init_vector(float* vec, int len) {
	
	for (int i = 0; i < len; ++i) {
		vec[i] = i + 1;
	}

	float min = vec[0];
	float max = vec[len - 1];
	float dvalue = max - min;
	for (int i = 0; i < len; ++i) {
		vec[i] /= dvalue;
	}	
}

void init_vector1(float* vec, int len) {
	
	for (int i = 0; i < len; ++i) {
		vec[i] = 1;
	}

}

int main(int argc, char* argv[]) {
	
	const int len = 512;
	float* A = new float[len];
	float* B = new float[len];
	init_vector1(A, len);
	init_vector1(B, len);


	const int loops = 10; 
	steady_clock::time_point start = steady_clock::now();
	for (int n = 0; n < loops; ++n) {
		float sum = avx2_dot_product(A, B, len);
	}
	
	steady_clock::time_point stop = steady_clock::now();
	milliseconds time = duration_cast<milliseconds>(stop - start);
	std::cout << "avx time:" << time.count() << "ms." << std::endl;

	start = steady_clock::now();
	for (int n = 0; n < loops; ++n) {
		float sum = vector_dot(A, B, len);
	}
	
	stop = steady_clock::now();
	time = duration_cast<milliseconds>(stop - start);
	std::cout << "vector dot time:" << time.count() << "ms." << std::endl;

	delete[] A;
	delete[] B;
	return 0;
}
