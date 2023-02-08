#include<iostream>
#include<stdlib.h>
#include<omp.h>
#define size 1000
using namespace std;
int A[size],R[size],F[size];

void enum_sort(int A[])
{
	#pragma omp parallel for
	for(int i=0;i<size;i++)
	{
		F[i] = 999;
	}
	#pragma omp parallel for
	for(int i=0;i<size;i++)
	{
		{
			int count = 0;
			for(int j=0;j<size;j++)
				if(A[i]>A[j])
					count++;
			R[i] = count;
		}
	}
	#pragma omp parallel for
	for(int i=0;i<size;i++)
	{
		#pragma omp critical
		{
		while(F[R[i]] == A[i])
		{
			R[i]++;
		}
		F[R[i]] = A[i];
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
        enum_sort(A);
	cout<<"The array after sorting is : ";
        for(int i=0;i<size;i++)
        {
                cout<<F[i]<<" ";
        }
        cout<<"\n";
        return 0;
}


