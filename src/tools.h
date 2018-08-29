#ifndef TOOLS_H_
#define TOOLS_H_	

#ifdef __cplusplus
	extern "C" {
#endif

float sum(const float* vector, int len);

float dot(const float* a, const float* b, int len);

void* register_func();
#ifdef __cplusplus
	};
#endif

#endif //TOOLS_H_
