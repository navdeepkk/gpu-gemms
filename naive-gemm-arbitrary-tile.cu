#include<iostream>
#include<cuda_runtime.h>
#include<cuda.h>
#include<device_launch_parameters.h>
#include "common.h"
#define DTYPE float
#define M 4096 
#define N 4096
#define K 4096
#define MBLOCK 32
#define NBLOCK 32
#define Mtile 128 // This will actually be the loop step of `i` loop.
#define Ntile 128 // This will actually be the loop step of `j` loop.

using namespace std;

__global__ void GEMM(DTYPE * a, DTYPE * b, DTYPE * c, int m, int n, int k){
  // Since the actual computation tile size is greater than than the thread
  // block tile size, therefore we want to find out what size of the output tile
  // is a register calculating.
  // Now each thread will compute an output tile of size (Mchunk, Nchunk).
  constexpr int Mchunk = Mtile / MBLOCK;
  constexpr int Nchunk = Ntile / NBLOCK;

  // Instead find the iteration of the original loop nest that maps to this
  // thread block here.
  // It is more elegant to map the iterations instead of row or col. At the end
  // it doesn't matter becuase the iterations actually determine which row or
  // col is it.
  // In this particular launch setup with thread block sizes of (32,32) and each
  // thread calculating one outptut element, the globalthreadId.x and
  // globalthreadId.y is actually the iterations we are looking for.
 
  // The Outer loops iteration beginning that this thread block tile
  // is responsible for.
  int i_iter_tile_base = blockIdx.y * Mtile;
  int j_iter_tile_base = blockIdx.x * Ntile;

  // The Inner loop iteration beginning that this thread block tile is
  // responsible for.
  int i_iter_thread_base = threadIdx.y * Mchunk;
  int j_iter_thread_base = threadIdx.x * Nchunk;

  // The Global index start that this thread is responsible for.
  int i = i_iter_tile_base + i_iter_thread_base;
  int j = j_iter_tile_base + j_iter_thread_base;

  // Allocate a Ctile in registers of dimensions (Mchunk, Nchunk).
  // Dont know if this actually goes into the resgisters as register file cannot
  // be indexed.
  DTYPE Cout[Mchunk * Nchunk];
  for(int i = 0; i < Mchunk; ++i){
    for(int j = 0; j < Nchunk; ++j){
      Cout[i * Nchunk + j] = 0.0f;
    }
  }
  
  // K dimension is sequential so this is not mapped to the gpu compute
  // heirarchy.
  for(int i_iter = i, ci = 0; i_iter < i + Mchunk; ++i_iter, ++ci){
    for(int j_iter = j, cj = 0; j_iter < j + Nchunk; ++j_iter, ++cj){
      for(int kk = 0; kk < k; ++kk){
	if(i_iter < m && j_iter < n){
	  Cout[ci * Nchunk + cj] += a[i_iter * k + kk] * b[kk * n + j_iter];
	}
      }
    }
  }
  
  // Write back the result to the output matrix.
  for(int ii = 0; ii < Mchunk; ++ii){
    for(int jj = 0; jj < Nchunk; ++jj){
      c[(i + ii) * n + (j + jj)] = Cout[ii * Nchunk + jj];
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
  check_cuda_error(cudaMalloc(&d_a, m * k * sizeof(DTYPE)));
  check_cuda_error(cudaMalloc(&d_b, k * n * sizeof(DTYPE)));
  check_cuda_error(cudaMalloc(&d_c, m * n * sizeof(DTYPE)));

  initMatrix(h_a, m , k);	
  initMatrix(h_b, k , n);	
  initMatrix(h_c_gpu_res, m , n);	

  check_cuda_error(cudaMemcpy(d_a, h_a, m * k * sizeof(DTYPE), cudaMemcpyHostToDevice));
  check_cuda_error(cudaMemcpy(d_b, h_b, k * n * sizeof(DTYPE), cudaMemcpyHostToDevice));

  dim3 block(NBLOCK, MBLOCK, 1);
  dim3 grid((n + Ntile - 1) / Ntile, (m + Mtile - 1) / Mtile, 1);

  GEMM<<<grid, block>>>(d_a, d_b, d_c, m , n, k);	
  
  check_cuda_error(cudaPeekAtLastError());
  check_cuda_error(cudaDeviceSynchronize());
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

  // The Global index start that this thread is responsible for.cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
