//  nvcc -O3 -std=c++11 -use_fast_math -ccbin g++ -arch=compute_75 -code=sm_75 -expt-relaxed-constexpr 
//  Performs matrix mutliplication using shared memory tiles where ewach thread
//  may need to calculate and move more than one data element. Assumes matrices
//  stored in row major order. The loop structure followed is as(one level
//  tiling)
//  
//  for(int i = 0; i < M; i += Mtile){			      //Inter-Tile
//    for(int j = 0; j < N; j += Ntile){		      //Inter-Tile
//	for(int k = 0; k < K; k += Ktile){		      //Inter-Tile
//	  for(int ii = i; ii < i + Mtile; ++ii){	      //Intra-Tile
//	    for(int jj = j; jj < j + Ntile; ++jj){	      //Intra-Tile
//	      for(int iii = ii; iii < ii + Mchunk; ++iii){    //Per-Thread
//		for(int jjj = jj; jjj < jj + Nchunk; ++jjj){  //Per-Thread
//		  for(int kk = k; kk < k + Ktile; ++kk){      //Intra-Tile
//		    //body
//		  }
//		}
//	      }
//	    }
//	  }
//	}
//    }
//  }

#include<iostream>
#include<cuda_runtime.h>
#include<cuda.h>
#include<device_launch_parameters.h>
#include "common.h"
#define DTYPE float
#define M 1024 
#define N 1024 
#define K 1024 
#define MBLOCK 32
#define NBLOCK 32
#define Mtile 128  // This will actually be the loop step of `i` loop.
#define Ntile 128  // This will actually be the loop step of `j` loop.
#define Ktile 32  // This will actually be the loop step of `k` loop.

using namespace std;

__global__ void GEMM(DTYPE * a, DTYPE * b, DTYPE * c, int m, int n, int k){
  // Reserve shared memory tiles if to put in the operands.
  __shared__ DTYPE asmem[Mtile * Ktile]; 
  __shared__ DTYPE bsmem[Ktile * Ntile];

  // Since the actual computation tile size is greater than than the thread
  // block tile size, therefore we want to find out what size of the output tile
  // is a register calculating.
  // Now each thread will compute an output tile of size (Mchunk, Nchunk).
  constexpr int Mchunk = Mtile / MBLOCK;
  constexpr int Nchunk = Ntile / NBLOCK;

  // Calculate the chunk of data each thread has to copy from the global memroy
  // to shared memeory. It is equal to the total number of data elements in a
  // Tile / total number of threads in a thread block.
  // TODO: Insert checks here to see if the if the tile size in elements is less than
  // the number of threads in a thread block.
  constexpr int Achunktocopy = (Mtile * Ktile) / (MBLOCK * NBLOCK);
  constexpr int Bchunktocopy = (Ktile * Ntile) / (MBLOCK * NBLOCK);

  // Find the iteration of the original loop nest that maps to this thread
  // block here.
  // It is more elegant to map the iterations instead of row or col. At the end
  // it doesn't matter becuase the iterations actually determine which row or
  // col is it.
  // In this particular launch setup with thread block sizes of (32, 32) and each
  // thread calculating one outptut element, the globalthreadId.x and
  // globalthreadId.y is actually the iterations we are looking for.

  // The Outer loops iteration beginning that this thread block tile
  // is responsible for. These coordinates also marks the beginning of the
  // address a thread block needs to copy form the global memory to shared 
  // memory. This represents the coordinates in the data space not in the GPU
  // (processor) space.
  int i_iter_tile_base = blockIdx.y * Mtile; // Maps to inter-tile `i`.
  int j_iter_tile_base = blockIdx.x * Ntile; // Maps to inter-tile `j`.

  // The Inner loop iteration beginning that this thread block tile is
  // responsible for.
  int i_iter_thread_base = threadIdx.y * Mchunk;
  int j_iter_thread_base = threadIdx.x * Nchunk;

  // The Global index start that this thread is responsible for computing. It
  // will caluclate (Mchunk, Nchunk) starting from these indexes.
  int i = i_iter_tile_base + i_iter_thread_base;
  int j = j_iter_tile_base + j_iter_thread_base;

  // Linear thread id in the thread block.
  int linear_tid = (threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
  // Number of threads required to copy one row of A.
  int num_threads_to_copy_one_Arow = Ktile / Achunktocopy;
  // Number of threads required to copy one row of B.
  int num_threads_to_copy_one_Brow = Ntile / Bchunktocopy;

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
  // heirarchy. Inter tile K-loop
  for(int kk = 0; kk < k; kk += Ktile){
    // Base address in global tile of A operand.
    int A_tile_base_addr = i_iter_tile_base * k + kk;
    // Base address in global tile of B operand.
    int B_tile_base_addr = kk * n + j_iter_tile_base;

    int A_row_to_copy_in_global = linear_tid / num_threads_to_copy_one_Arow;
    int A_col_to_copy_in_global = linear_tid % num_threads_to_copy_one_Arow * Achunktocopy; 
    int B_row_to_copy_in_global = linear_tid / num_threads_to_copy_one_Brow;
    int B_col_to_copy_in_global = linear_tid % num_threads_to_copy_one_Brow * Bchunktocopy; 

    // Copy the operands from global to shared memory. Each thread copies the
    // `chunktocopy` elements from global to sharedm memory. The thread Id's
    // inside a thread block need to be linearized. Each thread copies it's
    // contiguous chunk form global memory to the shared memroy.
    #pragma unroll
    for(int cpi = 0; cpi < Achunktocopy; ++cpi){
      asmem[linear_tid * Achunktocopy + cpi] = a[A_tile_base_addr + (A_row_to_copy_in_global * k) + 
	A_col_to_copy_in_global + cpi]; 
    }
    #pragma unroll
    for(int cpi = 0; cpi < Bchunktocopy; ++cpi){
      bsmem[linear_tid * Bchunktocopy + cpi] = b[B_tile_base_addr + (B_row_to_copy_in_global * k) + 
	B_col_to_copy_in_global + cpi]; 
    }
    __syncthreads();
    // Start the computation using fast memory buffers.
    // This is the amount of work done by one thread i.e., computaion of one
    // (Mchunk, Nchunk) tile in output.
    // Do not use the global thread indices `i` and `j` here. We only need the
    // thread info inside a thread block and hence we need to start it with.
    // Evertyhing changes here `i_iter`, `j_iter`, `k_iter` were being used to
    // denote the global iteration that was taking place but now since things
    // have to be indexed in the shared memory now we cannot use `i_iter`,
    // `j_iter` and `k_iter` to index them. Now `i_iter` and `j_iter` is set to
    // use the thread identifier within the thread block. `k_iter` is set to
    // start from zero and then go upto `ktile`.
    #pragma unroll
    for(int i_iter = i_iter_thread_base, ci = 0; i_iter < i_iter_thread_base + Mchunk; ++i_iter, ++ci){
      #pragma unroll
      for(int j_iter = j_iter_thread_base, cj = 0; j_iter < j_iter_thread_base + Nchunk; ++j_iter, ++cj){
	// Intra-tile K-loop.
	#pragma unroll
	for(int k_iter = 0; k_iter < Ktile; ++k_iter){
	  //printf("i:%d, j:%d, k:%d\n", i_iter, j_iter, k_iter);
	  if(i_iter < Mtile && j_iter < Ntile){
	    // This statement now uses the shared memory fast buffers.
	    Cout[ci * Nchunk + cj] += asmem[i_iter * Ktile + k_iter] * bsmem[k_iter * Ntile + j_iter];
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
    __syncthreads();
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

  //printf("%d, %d, %d\n", block.x, block.y, block.z);
  //printf("%d, %d, %d\n", grid.x, grid.y, grid.z);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  GEMM<<<grid, block>>>(d_a, d_b, d_c, m , n, k);	
  cudaEventRecord(stop, NULL);

  cudaEventSynchronize(stop);
  float msecTotal = 0.0f;
  cudaEventElapsedTime(&msecTotal, start, stop);
  double flopsPerMatrixMul = 2.0 * (double) m * (double) n * (double) k;
  double teraFlops = (flopsPerMatrixMul * 1.0e-12f) / (msecTotal / 1000.0f);
  cout<<"PERF: "<<teraFlops<<"Tflops \n";

  check_cuda_error(cudaPeekAtLastError());
  check_cuda_error(cudaDeviceSynchronize());
  cudaMemcpy(h_c_gpu_res, d_c, m * n * sizeof(DTYPE), cudaMemcpyDeviceToHost);
  //hostGEMM(h_a, h_b, h_c, m, n, k);

  //if(compareGEMM(h_c, h_c_gpu_res, m, n))
  //  cout<<"Success!\n";
  //else
  //  cout<<"Output does not amtch!\n";

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
