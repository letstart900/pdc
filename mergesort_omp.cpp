#include<iostream>
#include<stdlib.h>
#include<omp.h>
#define size 5
using namespace std;
int A[size];

void merge(int A[],int l, int mid, int r)
{
	int i, j, k;
	i = l;
	k = l;
	j = mid+1;
	int R[size];

	while(i<=mid && j<=r)
	{
		if(A[i]>A[j])
		{
			R[k] = A[j];
			j++;
		}
		else
		{
			R[k] = A[i];
			i++;
		}
		k++;
	}
	while(i<=mid)
	{
		R[k] = A[i];
		i++;
		k++;
	}
	while(j<=r)
	{
		R[k] = A[j];
		j++;
		k++;
	}
	for(int i=l;i<=r;i++)
	{
		A[i] = R[i];
	}
}

void mergesort(int A[], int l, int r)
{
	if(l<r)
	{
		int mid = (l+r)/2;
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				mergesort(A,l,mid);
			}
			#pragma omp section
			{
				mergesort(A,mid+1,r);
			}
		}
		merge(A,l,mid,r);
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
	mergesort(A,0,size-1);
	cout<<"The array after sorting is : ";
        for(int i=0;i<size;i++)
        {
                cout<<A[i]<<" ";
        }
        cout<<"\n";
        return 0;
}
