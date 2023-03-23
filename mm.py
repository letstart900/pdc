%%cu
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

#define M 5120
#define K 64
#define N 5120

#define BLOCK_SIZE 32

using namespace std;

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cout << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cout << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
    else {
        cout<<"\nNO CUDA ERROR";
    }
}

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

void printMatrix(vector<int> matrix, vector<int> dims, char var_name)
{
    cout<<endl<<var_name<<" : ("<<dims[0]<<", "<<dims[1]<<")"<<endl;
    for(int i = 0; i < dims[0]; i++)
    {
        for(int j = 0; j < dims[1]; j++)
        {
            cout<<matrix[i* dims[1] + j]<<" ";
        }
       cout<<endl;
    }
}

__global__ void matrixMultiplication(int *matrix1, int *dim1, int *matrix2, int *dim2, int *result, int *dim3)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < dim3[0] && col < dim3[1])
    {
        int sum = 0;
        for(int z = 0; z < dim1[1]; z++)
        {
            sum += matrix1[row * dim1[1] + z] * matrix2[z * dim2[1] + col];
        }
        result[row * dim3[1] + col] = sum;
    }

}

void matrixMultiplicationCPU(int *matrix1, int *dim1, int *matrix2, int *dim2, int *result, int *dim3)
{
    for(int i = 0; i < dim3[0]; i++)
    {
        for(int j = 0; j < dim3[1]; j++)
        {
            int sum = 0;
            for(int k = 0; k < dim1[1]; k++)
            {
                sum += matrix1[i * dim1[1] + k] * matrix2[k * dim2[1] + j];
            }
            result[i * dim3[1] + j] = sum;
        }
    }
}

vector<int> getMatrix(int size)
{
    vector<int> matrix;
    for(int i = 0; i < size; i++)
    {
        matrix.push_back(i + 1);
    }
    return matrix;
}

double absError(vector<int> values1, vector<int> values2)
{
    int size = values1.size();
    double ans = 0.0;
    for(int i = 0; i < size; i++)
    {
        ans += abs(values1[i] - values2[i]);
    }
    return ans;
}

int main()
{
    
    dim3 blockDimensions(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDimensions(M/BLOCK_SIZE, N/BLOCK_SIZE);
 
    vector<int> matrix1 = getMatrix(M * K);
    vector<int> matrix2 = getMatrix(K * N);
    vector<int> result(M * N, 0);
 
    vector<int> dimension1 = {M, K};
    vector<int> dimension2 = {K, N};
    vector<int> dimension3 = {M, N};

//    printMatrix(matrix1, dimension1, 'A');
//    printMatrix(matrix2, dimension2, 'B');
    
    int *d_m1, *d_m2, *d_m3; // matrix 
    int *d_d1, *d_d2, *d_d3; // dimensions of each matrix
 
    cudaMalloc((void **)&d_m1, M * K * sizeof(int));
    cudaMalloc((void **)&d_m2, K * N * sizeof(int));
    cudaMalloc((void **)&d_m3, M * N * sizeof(int));
 
    cudaMalloc((void **)&d_d1, 2 * sizeof(int));
    cudaMalloc((void **)&d_d2, 2 * sizeof(int));
    cudaMalloc((void **)&d_d3, 2 * sizeof(int));
 
    cudaMemcpy(d_m1, matrix1.data(), M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, matrix2.data(), K * N * sizeof(int), cudaMemcpyHostToDevice);
 
    cudaMemcpy(d_d1, dimension1.data(), 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d2, dimension2.data(), 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d3, dimension3.data(), 2 * sizeof(int), cudaMemcpyHostToDevice);
 

    startTimer();
    matrixMultiplication<<<gridDimensions, blockDimensions>>>(d_m1, d_d1, d_m2, d_d2, d_m3, d_d3);
    cudaDeviceSynchronize();
    double timeTakenGPU = stopTimer();
    CHECK_LAST_CUDA_ERROR();
 

    cout<<"\nGPU Took : "<<timeTakenGPU<<endl;
    cudaMemcpy(result.data(), d_m3, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    vector<int> copyVec(result);
//    printMatrix(result, dimension3, 'G');

    startTimer();
    matrixMultiplicationCPU(matrix1.data(), dimension1.data(), matrix2.data(), dimension2.data(), result.data(), dimension3.data());
    double timeTakenCPU = stopTimer();
    cout<<"\nCPU Took : "<<timeTakenCPU<<endl;
//    printMatrix(result, dimension3, 'C');
    cout<<"\nComputational Error: "<<absError(copyVec, result);
	  cout<<"\nSpeedup is "<<timeTakenCPU/timeTakenGPU<<endl;


    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_m3);
    cudaFree(d_d1);
    cudaFree(d_d2);
    cudaFree(d_d3);

    return 0;
}
