%%writefile parallelPrefix.cu
#include <stdio.h>

#define BLOCK_SIZE 1024
#define N 1024

void printArray(int *arr, int size)
{
  for(int i = 0; i < size; i++)
  {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

void errorCheck()
{
  cudaError_t code = cudaPeekAtLastError();
  printf("%s", cudaGetErrorString(code));
}

__global__ void prefixSum(int *d_arr, int *output)
{
  int gid = threadIdx.x + blockDim.x * blockIdx.x;

  if(gid < N)
  {
    int copyArr[N];

    for(int d = 1; d < N; d *= 2)
    {
      copyArr[gid] = d_arr[gid];
      __syncthreads();
      if(gid >= d)
      {
        d_arr[gid] = copyArr[gid] + d_arr[gid - d];
      }
      __syncthreads();
    }

    output[gid] = d_arr[gid];
  }

}

int *parallelPrefix(int *array)
{
  size_t bytes = N * sizeof(int);
  int *d_array, *output;

  cudaMalloc((void **)&d_array, bytes);
  cudaMalloc((void **)&output, bytes);

  cudaMemcpy(d_array, array, bytes, cudaMemcpyHostToDevice);
  
  int blockSize = min(N, BLOCK_SIZE);
  dim3 gridDim((N + blockSize - 1)/blockSize);
  dim3 blockDim(blockSize);

  prefixSum <<< gridDim, blockDim>>> (d_array, output);
  cudaDeviceSynchronize();
  errorCheck();


// dynamic memory allocation wont be deallocated after function ends
  int *prefix = (int *) malloc(bytes);
  cudaMemcpy(prefix, output, bytes, cudaMemcpyDeviceToHost);
  return prefix;

}

void fillPrefixSum(int arr[], int n, int prefixSum[])
{
	prefixSum[0] = arr[0];
	for (int i = 1; i < n; i++)
		prefixSum[i] = prefixSum[i - 1] + arr[i];
}

void verify(int *values1, int *values2)
{
  for(int i = 0; i < N; i++)
  {
    if(values1[i] != values2[i])
    {
      printf("NOT EQUAL\n");
      return;
    }
  }
  printf("NO ERROR\n");
}

int main()
{
  int array[N];
  for(int i = 0; i < N; i++) array[i] = i + 1;
  int *prefix = parallelPrefix(array);
  printf("\nPREFIX\n");
  printArray(prefix, N);

  int seqPrefix[N];
  fillPrefixSum(array, N, seqPrefix);
  verify(seqPrefix, prefix);
}

!nvcc parallelPrefix.cu
!./a.out

