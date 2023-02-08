#include<iostream>
#include<omp.h>
#define size 5
int A[size];
using namespace std;

void odd_even_sort(int A[],int n)
{
	bool sorted = false;

	while(!sorted)
	{
		sorted = true;
		#pragma omp parallel for
		{
		for(int i=1;i<=n-2;i+=2)
		{
			if(A[i]>A[i+1])
			{
				int temp = A[i];
				A[i] = A[i+1];
				A[i+1] = temp;
				sorted = false;
			}
		}
		}
		#pragma omp parallel for
		{
		for(int i=0;i<=n-2;i+=2)
		{
			if(A[i]>A[i+1])
			{
				int temp = A[i];
				A[i] = A[i+1];
				A[i+1] = temp;
				sorted = false;
			}
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
        odd_even_sort(A,size);
        cout<<"The array after sorting is : ";
        for(int i=0;i<size;i++)
        {
                cout<<A[i]<<" ";
        }
        cout<<"\n";
        return 0;
}
