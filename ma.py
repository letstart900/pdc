%%cu
#include <iostream>
#include <chrono>
#include <vector>

#define BLOCK_SIZE 32
#define M 10240
#define N 10240

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


vector<int> getMatrix()
{
    vector<int> matrix;
    for(int i = 0; i < M * N; i++)
    {
        matrix.push_back(i+1);
    }
    return matrix;
}

void printMatrix(vector<int> matrix, char var)
{
    cout<<var<<":\n";
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            cout<<matrix[i * N + j]<<" ";
        }
        cout<<endl;
    }
}

__global__ void matrixAddition(int *matrix1, int *matrix2, int *result)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
 
    if(row < M && col < N)
    {
        result[row * N + col] = matrix1[row * N + col] + matrix2[row * N + col];
    }
}

void matrixAdditionCPU(int *matrix1, int *matrix2, int *result)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            result[i * N + j] = matrix1[i * N + j] + matrix2[i * N + j];
        }
    }
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
    size_t size = M * N * sizeof(int);
    vector<int> matrix1 = getMatrix();
    vector<int> matrix2 = getMatrix();
    vector<int> result(M * N, 0);
 
    int *d_m1, *d_m2, *d_m3;
    cudaMalloc((void **)&d_m1, size);
    cudaMalloc((void **)&d_m2, size);
    cudaMalloc((void **)&d_m3, size);

    cudaMemcpy(d_m1, matrix1.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, matrix2.data(), size, cudaMemcpyHostToDevice);
 
//    printMatrix(matrix1, 'A');
//    printMatrix(matrix2, 'B');
 

    dim3 blockDims(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDims(M/BLOCK_SIZE, N/BLOCK_SIZE);
 
    double timeTakenGPU;
    startTimer();
    matrixAddition<<<gridDims, blockDims>>>(d_m1, d_m2, d_m3);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    timeTakenGPU = stopTimer();
    cout<<"\nGPU Time: "<<timeTakenGPU<<endl;
    cudaMemcpy(result.data(), d_m3, size, cudaMemcpyDeviceToHost);
 
//    printMatrix(result, 'G');



    vector<int> resultCPU(M * N, 0);
    double timeTakenCPU;
    startTimer();
    matrixAdditionCPU(matrix1.data(), matrix2.data(), resultCPU.data());
    timeTakenCPU = stopTimer();
    cout<<"\nCPU Time: "<<timeTakenCPU<<endl;
    cout<<"\nComputational Error: "<<absError(resultCPU, result);
    cout<<"\nSpeedup: "<<timeTakenCPU/timeTakenGPU<<endl;
//    printMatrix(resultCPU, 'C');

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_m3);
    
}
