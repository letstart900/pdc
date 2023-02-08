#include<iostream>
#include<stdlib.h>
#include<omp.h>
#define size 5
using namespace std;
int A[size];

int partition(int arr[],int start,int end)
{
        int i=start-1;
        int p=end;
        for(int j=start;j<end;j++)
        {
                if(arr[j]<arr[p])
                {
                        i++;
                        int temp = arr[j];
                        arr[j] = arr[i];
                        arr[i] = temp;
                }
       }
        int temp=arr[i+1];
        arr[i+1]=arr[p];
        arr[p]=temp;
        return i+1;
}

void quicksort(int arr[], int start, int end)
{
        if(start<end)
        {
                int p = partition(arr,start,end);
                #pragma omp parallel sections
                {
                        #pragma omp section
                        {
                                quicksort(arr,start,p-1);
                        }
                        #pragma omp section
                        {
                                quicksort(arr,p+1,end);
                        }
                }
        }
}

int main()
{
        srand(time(0));
        cout<<"The array before sorting is : ";
        for(int i=0;i<size;i++)
        {
                A[i]=rand()%10;
                cout<<A[i]<<" ";
        }
        cout<<"\n";
        quicksort(A,0,size-1);
        cout<<"The array after sorting is : ";
        for(int i=0;i<size;i++)
        {
                cout<<A[i]<<" ";
        }
        cout<<"\n";
        return 0;
}
