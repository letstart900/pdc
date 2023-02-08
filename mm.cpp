#include<iostream>
#include<omp.h>
#include<string.h>
#define size 4
using namespace std;
int A[size][size], B[size][size], C[size][size];
int main()
{
	cout<<"Matrix Multiplication::\n";
	srand(time(0));
	for(int i=0;i<size;i++)
		for(int j=0;j<size;j++)
		{
			A[i][j] = rand()%10;
			B[i][j] = rand()%10;
		}
	cout<<"Matrix A::\n";
	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
		{
			cout<<A[i][j]<<" ";
		}
		cout<<"\n";
	}
	cout<<"Matrix B::\n";
	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
		{
			cout<<B[i][j]<<" ";
		}
		cout<<"\n";
	}
	int sum;
	for(int i=0;i<size;i++)
	{
		#pragma omp parallel private(sum)
		{
			sum=0;
			int threadId = omp_get_thread_num();
			#pragma omp for
			for(int j=0;j<size;j++)
			{
				for(int k=0;k<size;k++)
				{
					sum = sum + A[i][k] * B[k][j];
				}
				C[i][j] = sum;
			}
		}
	}

	cout<<"Resultant Matrix :: \n";
	for(int i=0;i<size;i++)
	{
		for(int j=0;j<size;j++)
		{
			cout<<C[i][j]<<" ";
		}
		cout<<"\n";
	}
	return 0;
}
