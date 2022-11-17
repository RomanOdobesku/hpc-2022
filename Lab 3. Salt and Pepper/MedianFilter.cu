#include "EasyBMP.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sys/times.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void kernel(int width, int height, float* arrOut, cudaTextureObject_t texObj) 
{
	int width_x = blockIdx.x * blockDim.x + threadIdx.x;
	int height_y = blockIdx.y * blockDim.y + threadIdx.y;
	int arr[9];
	int k = 0;
	if ((width_x < width) && (height_y < height))
	{
		for (int i = width_x - 1; i <= width_x + 1; i++)
			for (int j = height_y - 1; j <= height_y + 1; j++)
			{
				arr[k] = (int)tex2D<float>(texObj, i, j);
				k++;
			}
		for (int q = 0; q < 9; q++)
			for (int w = 0; w < 8; w++)
				if (arr[w] > arr[w + 1])
				{
					float b = arr[w];
					arr[w] = arr[w + 1];
					arr[w + 1] = b;
				}
		arrOut[(width_x)+(height_y) * (width)] = arr[4];
	}
}
void WriteProcent(int procent, long double time) {
	system("clear");
	cout << procent << "%" << endl;
	cout << "Time left: " << time << " sec" << endl;
}

vector<int> SortArray(vector<int> arr) {
	for (int q = 0; q < 9; q++)
		for (int w = 0; w < 8; w++)
			if (arr[w] > arr[w + 1])
			{
				float b = arr[w];
				arr[w] = arr[w + 1];
				arr[w + 1] = b;
			}
	return arr;
}

int medianFilter(vector<vector<int>> image, int verctical, int gorizontal) {
	vector<int> arr;
	
	for (int i = verctical - 1; i <= verctical + 1; i++)
		for (int j = gorizontal - 1; j <= gorizontal + 1; j++)
			arr.push_back(image[i][j]);
	arr = SortArray(arr);
	return arr[4];
}

vector<vector<int>> changeImage(vector<vector<int>> image) {
	vector<vector<int>> output(image.size(), vector <int>(image[0].size()));
	float step = (image.size()-2) / 100.0;
	struct timespec mt1, mt2;
	clock_gettime(CLOCK_REALTIME, &mt1);
	long double timeIteration = 0;
	cout << setprecision(2) << fixed;
	for (int i = 1; i < image.size() - 1; i++)
	{
		if (i == 2)
		{
			clock_gettime(CLOCK_REALTIME, &mt2);
			timeIteration = (1000000000 * (mt2.tv_sec - mt1.tv_sec) + (mt2.tv_nsec - mt1.tv_nsec))/ 1000000000.0;
		}
		else
			WriteProcent((int)(i / step), (image.size() - 2 - i)*timeIteration);		
		for (int j = 1; j < image[0].size() - 1; j++)
			output[i][j] = medianFilter(image, i, j);
	}
	cout << setprecision(7) << fixed;
	return output;
}

bool GetCudaErrors(cudaError_t cudaError, const char* message)
{
	if (cudaError != cudaSuccess) {
		cout << message << cudaGetErrorString(cudaError) << endl;
		return true;
	}
	return false;
}
int main()
{
	system("clear");
	BMP Input;
	Input.ReadFromFile("input.bmp");
	int width = Input.TellWidth();
	int height = Input.TellHeight();

	vector<vector<int>> imageCpu(width + 2, vector <int>(height + 2));

	for (int j = 0; j < height; j++)
		for (int i = 0; i < width; i++)
			imageCpu[i + 1][j + 1] = Input(i, j)->Red;

	float* imageGpu = (float*)malloc(width * height * sizeof(float));
	for (int i = 1; i < width + 1; ++i)
		for (int j = 1; j < height + 1; ++j)
			imageGpu[i * height + j] = imageCpu[i + 1][j + 1];

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray_t arrIn;
	float* arrOut;

	cudaError_t cuerr = cudaMalloc((void**)&arrOut, width * height * sizeof(float));
	if (GetCudaErrors(cuerr, "Cannot allocate device Ouput array for a: ")) return 0;

	cuerr = cudaMallocArray(&arrIn, &channelDesc, width, height);
	if (GetCudaErrors(cuerr, "Cannot allocate device Input array for a: ")) return 0;

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cuerr = cudaEventCreate(&start);
	if (GetCudaErrors(cuerr, "Cannot create CUDA start event: ")) return 0;

	cuerr = cudaEventCreate(&stop);
	if (GetCudaErrors(cuerr, "Cannot create CUDA end event: ")) return 0;

	cuerr = cudaMemcpy2DToArray(arrIn, 0, 0, imageGpu, (width) * sizeof(float), (width) * sizeof(float), (height), cudaMemcpyHostToDevice);
	if (GetCudaErrors(cuerr, "Cannot copy a array2D from host to device: ")) return 0;

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = arrIn;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0; 

	cudaTextureObject_t texObj = 0;
	cuerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	if (GetCudaErrors(cuerr, "Cannot create TextureObject: ")) return 0;

	dim3 BLOCK_SIZE(32, 32, 1);
	dim3 GRID_SIZE(height / 32 + 1, width / 32 + 1, 1);

	cuerr = cudaEventRecord(start, 0);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot record CUDA event: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}

	kernel << <GRID_SIZE, BLOCK_SIZE >> > (width, height, arrOut, texObj);

	cuerr = cudaGetLastError();
	if (GetCudaErrors(cuerr, "Cannot launch CUDA kernel: ")) return 0;

	cuerr = cudaDeviceSynchronize();
	if (GetCudaErrors(cuerr, "Cannot synchronize CUDA kernel: ")) return 0;

	cuerr = cudaMemcpy(imageGpu, arrOut, width * sizeof(float) * height, cudaMemcpyDeviceToHost);
	if (GetCudaErrors(cuerr, "Cannot copy a array from device to host: ")) return 0;

	for (int j = 0; j < height; j++)
		for (int i = 0; i < width; i++)
		{
			ebmpBYTE color = (ebmpBYTE)imageGpu[i * height + j];
			Input(i, j)->Red = color;
			Input(i, j)->Green = color;
			Input(i, j)->Blue = color;
		}

	cuerr = cudaEventRecord(stop, 0);
	if (GetCudaErrors(cuerr, "Cannot record CUDA event: ")) return 0;

	struct timespec mt1, mt2;
	long double tt;
	clock_gettime(CLOCK_REALTIME, &mt1);

	imageCpu = changeImage(imageCpu);

	clock_gettime(CLOCK_REALTIME, &mt2);
	tt = 1000000000 * (mt2.tv_sec - mt1.tv_sec) + (mt2.tv_nsec - mt1.tv_nsec);
	cout << "Time CPU: " << tt / 1000000000 << " second" << endl;

	cuerr = cudaEventElapsedTime(&gpuTime, start, stop);

	cout << "Width: " << width << endl;
	cout << "Height: " << height << endl;
	cout << "Time GPU: " << gpuTime / 1000 << " second" << endl;
	cout << "SpeedUp: " << tt / (gpuTime * 1000000) << endl;

	BMP Output;
	Output.ReadFromFile("input.bmp");

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			ebmpBYTE color = (ebmpBYTE)imageCpu[i + 1][j + 1];
			Output(i, j)->Red = color;
			Output(i, j)->Green = color;
			Output(i, j)->Blue = color;
		}
	}

	Input.WriteToFile("outputGPU.bmp");
	Output.WriteToFile("outputCPU.bmp");

	cudaDestroyTextureObject(texObj);
	cudaFreeArray(arrIn);
	cudaFree(arrOut);

	free(imageGpu);
	return 0;
}