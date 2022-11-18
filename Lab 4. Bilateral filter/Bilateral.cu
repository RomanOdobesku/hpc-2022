#include "EasyBMP.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sys/times.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void kernel(int width, int height, float* arrOut, cudaTextureObject_t texObj, float sigma, float* G) 
{
	int width_x = blockIdx.x * blockDim.x + threadIdx.x;
	int height_y = blockIdx.y * blockDim.y + threadIdx.y;
	float a0 = tex2D<float>(texObj, width_x, height_y);
	float hai = 0;
	float ai;
	int index = 0;
	float k = 0;
	if ((width_x < width) && (height_y < height))
	{
		for (int i = width_x - 1; i <= width_x + 1; i++)
			for (int j = height_y - 1; j <= height_y + 1; j++)
			{
				ai = tex2D<float>(texObj, i, j);
				float rai = exp((pow(ai - a0, 2.0)) / (pow(sigma, 2.0)));
				hai += ai * G[index] * rai;
				k += G[index] * rai;
				++index;				
			}
		arrOut[(width_x)+(height_y) * (width)] = (hai / k);
	}
}
void WriteProcent(int procent, long double time) {
	system("clear");
	cout << procent << "%" << endl;
	int hours = (int)(time/3600);
	int minutes = (int)((time - hours*3600)/60);
	time = time - hours*3600 - minutes*60;
	cout << "Time left: " << hours << ":" << minutes << ":" << time << " hh:mm:ss.ms " << endl;
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

void GetG(float sigma, float* cpuG) {
	int index = 0;
	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
			cpuG[index++] = exp((pow(i, 2) + pow(j, 2)) / (-1*pow(sigma,2)));
}

int BilateralFilter(int vert, int gor, float sigma, float* cpuG, vector<vector<int>> image) {
	double a0 = image[vert][gor];
	double hai = 0;
	double ai;
	int index = 0;
	double k = 0;
	for (int i = vert - 1; i <= vert + 1; i++)
	{
		for (int j = gor - 1; j <= gor + 1; j++)
		{
			ai = image[i][j];
			float rai = exp((pow(ai - a0, 2)) / (pow(sigma, 2)));
			k += cpuG[index] * rai;
			hai += ai * cpuG[index] * rai;
			++index;
		}
	}
	return (int)(hai / k);
}
vector<vector<int>> changeImage(float sigma, float* cpuG, vector<vector<int>> image) {
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
			output[i][j] = BilateralFilter(i, j, sigma, cpuG, image);
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
int main(int argc, char* argv[])
{
	system("clear");

	float sigmaD = atof(argv[1]);
  	float sigmaR = atof(argv[2]);
	if (sigmaD < 0 || sigmaR < 0 )
	{
		cout << "Check the sigmaD and sigmaR parameters" << endl;
		return 0;
	}

	float* cpuG = (float*)malloc(9 * sizeof(float));
	float* gpuG;

	GetG(sigmaD, cpuG);

	cudaError_t cuerr = cudaMalloc(&gpuG, 9 * sizeof(float));
	if (GetCudaErrors(cuerr, "Cannot allocate device Ouput array for G_GPU: ")) return 0;

	cuerr = cudaMemcpy(gpuG, cpuG , 9 * sizeof(float), cudaMemcpyHostToDevice);
	if (GetCudaErrors(cuerr, "Cannot copy a array from device to host: ")) return 0;

	BMP Input;
	Input.ReadFromFile("input.bmp");
	int width = Input.TellWidth();
	int height = Input.TellHeight();

	vector<vector<int>> imageCpu(width + 2, vector <int>(height + 2));
	float* imageGpu = (float*)malloc(width * height * sizeof(float));

	for (int j = 0; j < height; j++)
		for (int i = 0; i < width; i++)
			{
				int temp = (int)floor(0.3 * Input(i, j)->Red + 0.59 * Input(i, j)->Green + 0.11 * Input(i, j)->Blue);
				imageGpu[i * height + j] = temp;
				imageCpu[i + 1][j + 1] = temp;
			}
			
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray_t arrIn;
	float* arrOut;

	cuerr = cudaMalloc((void**)&arrOut, width * height * sizeof(float));
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

	kernel << <GRID_SIZE, BLOCK_SIZE >> > (width, height, arrOut, texObj, sigmaR, gpuG);

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

	imageCpu = changeImage(sigmaR, cpuG, imageCpu);

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