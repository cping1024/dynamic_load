#include <iostream>
#include <dlfcn.h>
#include <stdlib.h>
#include <vector>
#include <gperftools/profiler.h>

#define LIB_NAME "libTools.so"

#define FUNC_SUM_NAME "sum"
#define FUNC_DOT_NAME "dot"
#define FUNC_REGISTER_NAME "register_func"

typedef struct func_table {
        float (*sum)(const float*, int);
        float (*dot)(const float*, const float*, int);
} func_table;

int main(int argc, char* argv[]) {

	ProfilerStart("dynamic.prof");
	void* handle = dlopen(LIB_NAME, RTLD_NOW);
	if (!handle) {
		std::cout << "open library failed." << std::endl;
		return -1;
	}

	// clear error info
	dlerror();

	// test use sum function
	void* (*reg_fun)();
	*(void **)&reg_fun = dlsym(handle, FUNC_REGISTER_NAME);
	char* error = dlerror();
        if (error != NULL) {
		dlclose(handle);
        	std::cout << "look up symbol fail." << std::endl;
		return -1;
	}

	func_table* table = nullptr;
	if (reg_fun) {
		table = (func_table*)reg_fun();
	}

	if (!table) {
		dlclose(handle);
		std::cout << "read func table failed." << std::endl;
		return -1;
	}

	// test read symbol info
	Dl_info info;
	int ret = dladdr((void*)table->sum, &info);
	if (!ret) {
		std::cout << "read symbol info failed." << std::endl;
	} else {
		printf("info dli_fname[%s].\n", info.dli_fname);
		printf("info dli_sname[%s].\n", info.dli_sname);
	}

int cnt = 10;
for (int i = 0; i < cnt; ++i) {
	const int len = 256;
	std::vector<float> vector(len, i);
	float v_sum = table->sum(vector.data(), len);	
	std::cout << "vector sum:" << v_sum << std::endl;
}

	dlclose(handle); 
	ProfilerStop();
	return 0;
}
