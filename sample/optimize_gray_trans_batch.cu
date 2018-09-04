#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

int div1(int M, int N) { 
	return ((M - 1) / N + 1);
}

#define KESTREL_KERNEL_LOOP(i, n)                                                                \
        for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                         \
             i += blockDim.x * gridDim.x)

#define MAX_THREAD_IN_BLOCK (512)
#define KESTREL_KERNEL_CFG(total)                                                                \
        ((total + MAX_THREAD_IN_BLOCK - 1) / MAX_THREAD_IN_BLOCK), MAX_THREAD_IN_BLOCK

texture<unsigned char, 1, cudaReadModeElementType> texture1_;
texture<unsigned char, 2, cudaReadModeElementType> texture2_;

void __global__ gray_kernel1(unsigned char* data, float* output, int w, int h , int stride) {
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < w * h) {
		int _w = index % w;
		int _h = index / w;
		unsigned char* pixel = (unsigned char*)(data + (stride * _h  + _w * 3));
		output[index] = pixel[0] * 0.114 + pixel[1] * 0.587 + pixel[2] * 0.299;
	} 
}


__global__ void gray_kernel2(uint8_t *data, float *outdata, int32_t w, int32_t h, int32_t stride)
{
        KESTREL_KERNEL_LOOP(index, w * h)
        {
                int _h = index / w;
                int _w = index % w;
                const uint8_t *IMAGE = (uint8_t *)(data + stride * _h + 3 * sizeof(uint8_t) * _w);
                outdata[index] = IMAGE[0] * 0.114 + IMAGE[1] * 0.587 + IMAGE[2] * 0.299;
        }
}


__global__ void gray_kernel3(unsigned char* data, float* output, int w, int h, int stride) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x < w && y < h) {
		unsigned char* pixel = data + (y * stride + x * 3);
		output[y * w + x] = pixel[0] * 0.114 + pixel[1] * 0.587 + pixel[2] * 0.299;
	}
}

__global__ void gray_kernel4(unsigned char* data, float* output, int w, int h, int stride) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < w && y < h) {
		//unsigned char* pixel = tex1Dfetch(texture1_, stride * y + x * 3);
		output[y * w + x] = tex1Dfetch(texture1_, stride * y + x * 3) * 0.114 + tex1Dfetch(texture1_, stride * y + x * 3 + 1) * 0.587 + tex1Dfetch(texture1_, stride * y + x * 3 + 2) * 0.299;
		//output[y*w + x] = tex2D(texture1_, x, y) * 0.114 + tex2D(texture1_, x+1, y) * 0.587 + tex2D(texture1_, x+2, y) * 0.299; 
	}
}

__global__ void gray_kernel5(unsigned char* data, float* output, int w, int h, int stride) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = x * 3;
	if (x < w && y < h) {
		output[y * w + x] = tex2D(texture2_, index, y) * 0.114 + tex2D(texture2_, index + 1, y) * 0.587 + tex2D(texture2_, index + 2, y) * 0.299; 
	}
}

__global__ void gray_kernel6(unsigned char* data, float* output, int w, int h, int stride) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z;
	float scalar[] = {0.114f, 0.587f, 0.299f};
	int index = x * 3;
	if (x < w && y < h) {
		//output[y * w + x] = tex2D(texture2_, index, y) * 0.114 + tex2D(texture2_, index + 1, y) * 0.587 + tex2D(texture2_, index + 2, y) * 0.299; 
		output[y*w +x] += tex2D(texture2_, index + z, y) * scalar[z];
		//atomicAdd(output + (y*w + x), tex2D(texture2_, index + z, y) * scalar[z]);
	}
}

__global__ void gray_kernel7(unsigned char* data, float* output, int w, int h, int stride) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int n = blockIdx.z;

	int offset = n * stride * h + y * stride + x * 3;
	unsigned char b = tex1Dfetch(texture1_, offset);
	unsigned char g = tex1Dfetch(texture1_, offset + 1);
	unsigned char r = tex1Dfetch(texture1_, offset + 2);
	if (x < w && y < h) {
		output[w * h * n + y * w + x] = r * 0.114 + g * 0.587 + b * 0.299;
	}
}

int main(int argc, char* argv[]) {
	
	const std::string filename(argv[1]);
	const int batch_size = atoi(argv[2]);
	const int time =atoi(argv[3]);
	std::cout << "batch size:" << batch_size << std::endl;
	cv::Mat image = cv::imread(filename);
	if (!image.isContinuous()) {
		std::cout << "read image fail." << std::endl;
		return -1;
	}
		
	
	unsigned char* d_image_input = nullptr;
	float* d_image_gray = nullptr;

	const int w = image.cols;
	const int h = image.rows;
	const int stride = image.step;
	const int channel = image.channels();
	printf("input dims [%d, %d, %d, %d].\n", channel, w, h, stride);

	cudaMalloc((void**)&d_image_input, batch_size * w * h * channel * sizeof(char));
	cudaMalloc((void**)&d_image_gray, batch_size * w * h * sizeof(float));
	for (int i = 0; i < batch_size; ++i) {
		unsigned char* ptr = d_image_input + i * w * h * channel;
		cudaMemcpy(ptr, image.data, w * h * channel * sizeof(char), cudaMemcpyHostToDevice);
	}
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	const int block_thread = 16;
	cudaChannelFormatDesc channelDesc =
		     cudaCreateChannelDesc<unsigned char>();	
	for (int i = 0; i < time; ++i) {

		cudaBindTexture(NULL, texture1_, d_image_input, stride * h * batch_size);
		dim3 block(block_thread, block_thread, 1);
		dim3 grid(div1(w, block_thread), div1(h, block_thread), batch_size);
		gray_kernel7<<<grid, block>>>(d_image_input, d_image_gray, w, h, stride);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float used_time = 0.0f;
	cudaEventElapsedTime(&used_time, start, stop);

	int len = w * h;
	float *h_buffer = new float[len];
	for (int i = 0; i < batch_size; ++i) {
		float* ptr = d_image_gray + i * w * h;
		cudaMemcpy(h_buffer, ptr, w * h * sizeof(float), cudaMemcpyDeviceToHost);
		cv::Mat h_img(h, w, CV_32FC1, h_buffer);

		cv::Mat gray_img;
		h_img.convertTo(gray_img, CV_8U);
		char name[128];
		sprintf(name, "picture%d", i);
		//cv::imshow(std::string(name), gray_img);
		//cv::waitKey();	
	}

	std::cout << "cuda kernel run time:" << used_time << "ms" << std::endl;
	delete []h_buffer;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_image_input);
	cudaFree(d_image_gray);
	return 0;
}
