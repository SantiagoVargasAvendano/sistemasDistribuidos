#include <stdio.h>
#include <vector>

#pragma comment(lib , "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#define blocksize 8;

void matrix_read(double *L, int dimension){
    FILE *fp;
    int row, col;

    fp = fopen("matrix1000.txt", "r");

    if(fp == NULL){
        printf("Ayuda puta\n");
        return;
    }
    for(row = 0; row < dimension; row++){
        for(col = 0; col < dimension; col++){
            if(fscanf(fp, "%lf,", &L[row * dimension + col]) == EOF) 
                break;
            
        }
        if(feof(fp)) break;
    }

    fclose(fp);
}

__global__ void normalize(double *A, double *I, int n, int i){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < n && y < n){
        if(x == i){
            I[x*n + y] /= A[i*n + i];
            A[x*n + y] /= A[i*n + i];
        }
    }
}

__global__ void gaussJordan(double *A, double *I, int n, int i){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < n && y < n){
        if(x != i){
            I[x*n + y] -= I[i*n + y] * A[x*n + i];
            if(y != i){
                A[x*n + y] -= A[i*n + y] * A[x*n + i];   
            }
        }
    }
}

__global__ void set_zero(double *A, double *I, int n, int i){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < n && y < n){
        if(x != i){
            if(y == i){
                A[x*n + y] = 0;   
            }
        }
    }
}

void saveToFile(char *filename, double *A, int n, int h){

    FILE *ofile;
    ofile = fopen(filename, "w");
    for(int i = 0; i < h; i++){
        for(int j = 0; j< h; j++){
            fprintf(ofile,"%f\t", A[i*n + j]);
        }
        fprintf(ofile,"\n");
    }
    fclose(ofile);
}


void printMatrix(double *A, int n){
	int x, y;
	for(y = 0; y < n; y++){
		printf("\n");
		for(x = 0; x < n; x++){
		    printf("%f ",  A[y*n + x]);
		}
	}
	printf("\n");
}

int main(){

    const int n = 1000;

    double *iL = new double[n*n];
    double *L = new double[n*n];
    matrix_read(L, n);

    printf("Matriz inversa\n");
    double *d_A, *I, *d_I;
    float time;
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int ddsize = n*n*sizeof(double);

    dim3 threadsPerBlock(8, 8); // blocksize = 8
    dim3 numBlocks(125 , 125); // (n + blocksize -1) / blocksize

    err = cudaMalloc((void**)&d_A, ddsize);
    if(err != cudaSuccess){ fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err)); }
    err = cudaMalloc((void**)&d_I, ddsize);
    if(err != cudaSuccess){ fprintf(stderr, "Failed to allocate device vector I (error code %s)!\n", cudaGetErrorString(err)); }
    I = new double[n*n];

    for(int i=0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(i == j) I[i*n + j] = 1.0;
            else I[i*n + j] = 0.0;
        }
    }

    err = cudaMemcpy(d_A, L, ddsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){ fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err)); }
    err = cudaMemcpy(d_I, I, ddsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){ fprintf(stderr, "Failed to copy vector I from host to device (error code %s)!\n", cudaGetErrorString(err)); }

    saveToFile("mat.txt" ,L, n, n);

    cudaEventRecord(start, 0);


    for(int i= 0; i < n; i++){
        normalize <<<numBlocks, threadsPerBlock >>> (d_A, d_I, n, i);
        gaussJordan <<<numBlocks, threadsPerBlock >>> (d_A, d_I, n, i);
        set_zero <<<numBlocks, threadsPerBlock >>> (d_A, d_I, n, i);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    err = cudaMemcpy(iL, d_I, ddsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){ fprintf(stderr, "Failed to copy vector IL from host to device (error code %s)!\n", cudaGetErrorString(err)); }
    err = cudaMemcpy(I, d_A, ddsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){ fprintf(stderr, "Failed to copy vector d_A to I from host to device (error code %s)!\n", cudaGetErrorString(err)); }

    printf("Cuda time: %lf ms\n", time);
    saveToFile("inversa_cuda.txt" ,iL, n, n);

    cudaFree(d_A);
    cudaFree(d_I);

    
    double *c = new double[n*n];
    for(int i= 0 ; i< n; i++){
        for(int j = 0; j < n; j++){
            c[i*n + j] = 0;
            for(int x = 0; x < n; x++){
                c[i*n + j] = c[i*n + j] + L[i*n +x] * iL[x*n + j];
            }
        }
    }
    saveToFile("c.txt" ,c, n, n);

    return 0;
}