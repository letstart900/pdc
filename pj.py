%%writefile pj.cu
#include <bits/stdc++.h>
#include <chrono>

#define LENGTH_OF_LIST 1024

using namespace std;

chrono::time_point<chrono::system_clock> startTime, endTime;


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
        cout<<"\nNO CUDA ERROR"<<endl;
    }
}

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

__global__ void distanceInit(int *next, int *distance){
    
  int index = threadIdx.x;
  
  if( next[index] == 0 ){
    distance[index] = 0;
  }
  else{
    distance[index] = 1;      
  }
} 


__global__ void distanceIterator(int *next, int *distance, int *devCopyNext, int *devCopyDistance){
  
  int index = threadIdx.x;
  //int index = threadIdx.x + blockIdx.x * blockDim.x;

  while(next[index]!=0){
      
      //printf("c: threadId : %d, next[index] : %d, distance[index] : %d \n", index, next[index], distance[index]);

      devCopyNext[index] = next[index];
      devCopyDistance[index] = distance[index];
      __syncthreads();

      distance[index] = devCopyDistance[index] + devCopyDistance[devCopyNext[index]];
      next[index] = devCopyNext[devCopyNext[index]];
      __syncthreads();
  }
}


void fillArray(int *arr, int size){
    
  for(int i=0; i<size; i++){
    arr[i] = rand()%10;      
  }
}


void printArray(int *arr, int size){

  cout<<endl;    
  for(int i=0; i<size; i++){
    cout<<arr[i]<<" ";   
  }
  cout<<endl;
}


void sequentialImplementation(int *next, int *distance){
    
  int index = 0;
  while(next[index] != 0){
    index++;
  }

  int value_holder = 0;
  for(int i=index;i>=0;i--){
      
    distance[i] = value_holder;
    value_holder++;   
  }
}


float getError(int *a, int *b){
    
  int wrongs = 0;
  for(int i=0;i<LENGTH_OF_LIST;i++){
    if(a[i]!=b[i])
      wrongs++;
  }
  return wrongs;
}


int main(){

  int *val_arr = new int[LENGTH_OF_LIST];
  
  fillArray(val_arr, LENGTH_OF_LIST);
  cout<<"Value array: ";
  printArray(val_arr, LENGTH_OF_LIST);

  int *dev_val_arr;
  size_t size = LENGTH_OF_LIST * sizeof(int);  

  cudaMalloc((void **)&dev_val_arr, size);
  cudaMemcpy(dev_val_arr, val_arr, size, cudaMemcpyHostToDevice);

  
  int *next_arr = new int[LENGTH_OF_LIST];

  for(int i=1; i<LENGTH_OF_LIST; i++){
    next_arr[i-1] = i;
  }
  next_arr[LENGTH_OF_LIST-1] = 0;
  cout<<"Successor array: ";
  printArray(next_arr, LENGTH_OF_LIST);

  int *dev_next_arr;
  cudaMalloc((void **)&dev_next_arr, size);
  cudaMemcpy(dev_next_arr, next_arr, size, cudaMemcpyHostToDevice);

  int* distances = new int[LENGTH_OF_LIST];

  int* dev_distances;
  cudaMalloc((void **)&dev_distances, size);
  cudaMemcpy(dev_distances, distances, size, cudaMemcpyHostToDevice);

  startTimer();
  distanceInit<<<1 , LENGTH_OF_LIST>>>(dev_next_arr, dev_distances);
  cudaMemcpy(distances, dev_distances, size, cudaMemcpyDeviceToHost);

  double timeTakenGPU = stopTimer();

  cout<<"Distances Init:";
  printArray(distances, LENGTH_OF_LIST);

  startTimer();

  int *devCopyNext;
  cudaMalloc((void **)&devCopyNext, size); 
  
  int *devCopyDistance;
  cudaMalloc((void **)&devCopyDistance, size); 

  distanceIterator<<<1, LENGTH_OF_LIST>>>(dev_next_arr, dev_distances, devCopyNext, devCopyDistance);
  CHECK_LAST_CUDA_ERROR();
  cudaMemcpy(distances, dev_distances, size, cudaMemcpyDeviceToHost);

  timeTakenGPU = stopTimer();

  cout<<"\n---\nParallel Output:";
  printArray(distances, LENGTH_OF_LIST);
  cout<<"\nGPU Took : "<<timeTakenGPU<<endl;

  startTimer();

  int *seq_distances = new int[LENGTH_OF_LIST]; 
  sequentialImplementation(next_arr, seq_distances);
  cout<<"\nSequential Output:";
  printArray(seq_distances, LENGTH_OF_LIST);
  cout<<"\n---\nError: "<<getError(seq_distances, distances)<<" items"<<endl;
  double timeTakenCPU = stopTimer();

  cout<<"\nCPU Took : "<<timeTakenCPU<<endl;  

}

!nvcc pj.cu

!./a.out
