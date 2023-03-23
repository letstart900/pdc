%%cu
#include <iostream>
#include <chrono>

#define N 102400

using namespace std;

chrono::time_point<chrono::system_clock> startTime, endTime;

void startTimer()
{
		startTime = chrono::system_clock::now();
}

double stopTimer()
{
		endTime = chrono::system_clock::now();
		chrono::duration<double> elapsedSeconds = endTime - startTime;
		return elapsedSeconds.count();
}

void printArray(int *a, char var_name)
{
		cout<<var_name<<" : ";
		for(int i = 0; i < N; i++)
		{
				cout<<a[i]<<" ";		
		}
		cout<<endl;
}

__global__ void vectorAdd(int *d_a, int *d_b, int *d_c)
{
		int index = blockIdx.x;
		if(index < N)
		{
				d_c[index] = d_a[index] + d_b[index];		
		}
}

void vectorAddCPU(int *a, int *b, int *c)
{
		for(int i = 0; i < N; i++)
		{
				c[i] = a[i] + b[i];
		}
}

void fillArray(int *a, int size)
{
		for(int i = 0; i < size; i++)
		{
				a[i] = i + 1;
		}
}

double absError(int *values1, int *values2, int size)
{
    double ans = 0.0;
    for(int i = 0; i < size; i++)
    {
        ans += abs(values1[i] - values2[i]);
    }
    return ans;
}

int main()
{
	// Host Variables
	int a[N];
	int b[N];
	fillArray(a, N);
	fillArray(b, N);
	int c[N];
	printArray(a, 'A');
	printArray(b, 'B');

	// Device Variables
	size_t size = N * sizeof(N);
	int *dev_a;
	int *dev_b;
	int *dev_c;
	cudaMalloc((void **)&dev_a, size);
	cudaMalloc((void **)&dev_b, size);
	cudaMalloc((void **)&dev_c, size);

	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

	startTimer();
	vectorAdd<<<N, 1>>>(dev_a, dev_b, dev_c);
	cudaDeviceSynchronize();
	double timeTakenGPU = stopTimer();
	cout<<"\nGPU Took : "<<timeTakenGPU<<endl;
	cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
//	printArray(c, 'G');
	
  int cpuC[N];
	startTimer();
	vectorAddCPU(a, b, cpuC);
	double timeTakenCPU = stopTimer();
//  printArray(c, 'C');
	cout<<"\nCPU Took : "<<timeTakenCPU<<endl;
  cout<<"\nComputational Error: "<<absError(c, cpuC, N);
	cout<<"\nSpeedup is "<<timeTakenCPU/timeTakenGPU<<endl;

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

}
