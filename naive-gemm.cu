#include<iostream>
#include<cuda_runtime.h>
#include<cuda.h>
#include<device_launch_parameters.h>
#include "common.h"
#define DTYPE float
#define M 1024
#define N 2048
#define K 1024
#define MBLOCK 32
#define NBLOCK 32

using namespace std;

__global__ void GEMM(DTYPE * a, DTYPE * b, DTYPE * c, int m, int n, int k){
  // Find out the actual row and column that this thread inside the thread block
  // maps to.
  // int row = blockIdx.y;
  // int col = blockDim.x;
  // Instead find the iteration of the original loop nest that maps to this
  // thread block here.
  // It is more elegant to map the iterations instead of row or col. At the end
  // it doesn't matter becuase the iterations actually determine which row or
  // col is it.
  // In this particular launch setup with thread block sizes of (32,32) and each
  // thread calculating one outptut element, the globalthreadId.x and
  // globalthreadId.y is actually the iterations we are looking for.
  int i_iter = blockIdx.y * blockDim.y + threadIdx.y;
  int j_iter = blockIdx.x * blockDim.x + threadIdx.x;

  // K dimension is sequential so this is not mapped to the gpu compute
  // heirarchy.
  c[i_iter * n + j_iter] = 0.0f;
  for(int kk = 0; kk < k; ++kk){
    if(i_iter < m && j_iter < n){
      c[i_iter * n + j_iter] += a[i_iter * k + kk] * b[kk * n + j_iter];
    }
  }
}

void hostGEMM(DTYPE * a, DTYPE * b, DTYPE * c, int m, int n, int k){
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j ){
      DTYPE temp = 0;
      for(int kk = 0; kk < k ; ++kk){
	temp += a[i * k + kk] * b[kk * n + j];
      }
      c[i * n + j] = temp;
    }
  }	
}

bool compareGEMM(DTYPE * h_c, DTYPE * h_c_gpu_res, int m, int n){	
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j ){
      if(abs(h_c[i * n + j] - h_c_gpu_res[i * n + j]) > 1e-4)
	return false;	
    }
  }
  return true;
}

void initMatrix(DTYPE * matrix, int m, int n){
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      matrix[i * n + j] = static_cast <DTYPE> (rand()) / static_cast <DTYPE> (RAND_MAX);
    }
  }
}

void printMatrix(DTYPE * matrix, int m, int n){
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      cout<<matrix[i * n + j]<<" ";
    }
    cout<<endl;
  }
  cout<<endl;
}

int main(){
  DTYPE *d_a, *d_b, *d_c, *h_a, *h_b, *h_c, *h_c_gpu_res;
  int m ,n, k;

  m = M;
  n = N;
  k = K;

  h_a = (DTYPE*) malloc(m * k * sizeof(DTYPE));
  h_b = (DTYPE*) malloc(k * n * sizeof(DTYPE));
  h_c = (DTYPE*) malloc(m * n * sizeof(DTYPE));
  h_c_gpu_res = (DTYPE*) malloc(m * n * sizeof(DTYPE));
  cudaMalloc(&d_a, m * k * sizeof(DTYPE));
  cudaMalloc(&d_b, k * n * sizeof(DTYPE));
  cudaMalloc(&d_c, m * n * sizeof(DTYPE));

  initMatrix(h_a, m , k);	
  initMatrix(h_b, k , n);	
  initMatrix(h_c_gpu_res, m , n);	

  cudaMemcpy(d_a, h_a, m * k * sizeof(DTYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, k * n * sizeof(DTYPE), cudaMemcpyHostToDevice);

  dim3 block(NBLOCK, MBLOCK, 1);
  dim3 grid((n + NBLOCK - 1) / NBLOCK, (m + MBLOCK - 1) / MBLOCK, 1);

  GEMM<<<grid, block>>>(d_a, d_b, d_c, m , n, k);	
  cudaDeviceSynchronize();
  cudaMemcpy(h_c_gpu_res, d_c, m * n * sizeof(DTYPE), cudaMemcpyDeviceToHost);
  hostGEMM(h_a, h_b, h_c, m, n, k);

  cout<<compareGEMM(h_c, h_c_gpu_res, m, n)<<endl;

  //printMatrix(h_c, m, n);
  //cout<<"output gpu\n";
  //printMatrix(h_c_gpu_res, m, n);

  free(h_a);
  free(h_b);
  free(h_c);
  free(h_c_gpu_res);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
