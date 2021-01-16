//  nvcc -O3 -std=c++11 -use_fast_math -ccbin g++ -arch=compute_75 -code=sm_75 -expt-relaxed-constexpr 
//  Performs matrix mutliplication using shared memory tiles where ewach thread
//  may need to calculate and move more than one data element. Assumes matrices
//  stored in row major order. The loop structure followed is as(one level
//  tiling)
//  
//  for(int i = 0; i < M; i += Mtile){				    //Inter-tb-Tile
//    for(int j = 0; j < N; j += Ntile){			    //Inter-tb-Tile
//	for(int k = 0; k < K; k += Ktile){			    //Inter-tb-Tile
//	  for(int ii = i; ii < i + Mtile; ii += WarpMtile){	    //Inter-warp-Tile
//	    for(int jj = j; jj < j + Ntile; jj += WarpNtile){	    //Inter-warp-Tile
//	      for(int kk = k; kk < k + Ktile; kk += WarpKtile){	    //Inter-warp-Tile
//		for(int iii = ii; iii < ii + WarpMtile; ++iii){	    //Intra-warp-Tile
//		  for(int jjj = jj; jjj < jj + WarpNtile; ++jjj){   //Intra-warp-Tile
//		    for(int kkk = kk; kkk < kk + WarpKtile; ++kkk){ //Intra-warp-Tile
//		      //body
//		    }
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
#include<cuda_fp16.h>
#include<mma.h> 
#include "common.h"
#define DTYPECD float
#define DTYPEAB __half
#define M 1024 
#define N 1024 
#define K 1024 
#define Mtile 128  // This will actually be the loop step of `i` loop.
#define Ntile 128  // This will actually be the loop step of `j` loop.
#define Ktile 32   // This will actually be the loop step of `k` loop.
#define WarpMtile 64 
#define WarpNtile 64
#define WarpKtile 32
#define WarpSize 32
#define WarpdimX 4
#define WarpdimY 8
#define WarpdimZ 1

// Seeing here that the thread block tile is going to calculate 128x128 output
// and one warp tile is to calculate 64x64 we only need 4 warps. i.e., we need to
// have 32 x 4 = 128 threads. A thread block of the kind 32 x 4 etc. is not possible.
// We need to have the thread block dimensions as multiples of 32 which 4 clearly isn't.
// so we need to launch 128 threads in one dimension only a whole lot of refactoring is
// needed.

using namespace std;

typedef struct {
  unsigned x;
  unsigned y;
  unsigned z;
} WarpIdx;

__global__ void GEMM(DTYPEAB * a, DTYPEAB * b, DTYPECD * c, int m, int n, int k){
  // Struct holding the geometrical coordinates of the warp.
  WarpIdx warpIdx;

  int numThreads = blockDim.x * blockDim.y * blockDim.z;
  int numWarpsInM = 1, numWarpsInN = 1, numWarps = numThreads / WarpSize;

  // Number of warps in the `M` and `N` dimension, of the thread block. If there are
  // sufficient number of warps for the `N` dimension then use them. If not assign all
  // of them to the `N` dimension. If excessive are present then assign the remaining
  // them to the `M` dimension.
  if(numWarps <= Ntile / WarpNtile){
    numWarpsInN = numWarps;
  }else{
    numWarpsInN = Ntile / WarpNtile;
    numWarpsInM = numWarps / numWarpsInN;
  }

  // Reserve shared memory tiles for the operands.
  __shared__ DTYPEAB asmem[Mtile * Ktile];
  __shared__ DTYPEAB bsmem[Ktile * Ntile];
  __shared__ DTYPEAB csmem[Mtile * Ntile];

  // Calculate the chunk of data each thread has to copy from the global memroy
  // to shared memeory. It is equal to the total number of data elements in a
  // Tile / total number of threads in a thread block.
  // TODO: Insert checks here to see if the if the tile size in elements is less than
  // the number of threads in a thread block.
  int Achunktocopy = (Mtile * Ktile) / numThreads;
  int Bchunktocopy = (Ktile * Ntile) / numThreads;

  // Linear thread id in the thread block.
  int linear_tid = (threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

  //printf("%d \n", linear_tid);
  // Linear warp id in the thread block.
  int linear_warpid = linear_tid / WarpSize;

  //printf("%d \n", linear_warpid);
  warpIdx.x = linear_warpid % numWarpsInN; 
  warpIdx.y = linear_warpid / numWarpsInN; 
  warpIdx.z = 1; 
  //printf("%d %d \n", warpIdx.y, warpIdx.x);

  // Find the iteration of the original loop nest that maps to this thread
  // block here.
  // It is more elegant to map the iterations instead of row or col. At the end
  // it doesn't matter becuase the iterations actually determine which row or
  // col is it.
  // The Outer loops iteration beginning that this thread block tile
  // is responsible for. These coordinates also marks the beginning of the
  // address a thread block needs to copy form the global memory to shared 
  // memory. This represents the coordinates in the data space not in the GPU
  // (processor) space.
  int i_iter_tile_base = blockIdx.y * Mtile; // Maps to inter-tile `i`.
  int j_iter_tile_base = blockIdx.x * Ntile; // Maps to inter-tile `j`.

  // The local thread id inside the warp. This id is relative to this warp only.
  int warpLocalTidY = (linear_tid % WarpSize) / WarpdimX;
  int warpLocalTidX = (linear_tid % WarpSize) % WarpdimX;

  //printf("%d %d \n", i_iter_thread_base, j_iter_thread_base);

  // Number of threads required to copy one row of A from global to shared memory.
  int num_threads_to_copy_one_Arow = Ktile / Achunktocopy;
  // Number of threads required to copy one row of B form global to shared memory.
  int num_threads_to_copy_one_Brow = Ntile / Bchunktocopy;

  DTYPECD *c_tb_tile_base = c;
  DTYPEAB *a_tb_tile_base = a;
  DTYPEAB *b_tb_tile_base = b;

  DTYPECD *c_tb_tile_offset = c_tb_tile_base + i_iter_tile_base * n + j_iter_tile_base;
  DTYPEAB *a_tb_tile_offset;
  DTYPEAB *b_tb_tile_offset;

  DTYPECD *c_warp_tile_base = c_tb_tile_offset;
  DTYPEAB *a_warp_tile_base;
  DTYPEAB *b_warp_tile_base;

  // c_warp_tile_offset will be in the global memory tile
  // while for A and B they will be in the shared memory.
  DTYPECD *c_warp_tile_offset;
  DTYPEAB *a_warp_tile_offset_compute;
  DTYPEAB *b_warp_tile_offset_compute;
  DTYPECD *c_thread_tile_base;
  DTYPEAB *a_thread_tile_base_copy;
  DTYPEAB *b_thread_tile_base_copy;
  DTYPEAB *a_thread_tile_base_compute;
  DTYPEAB *b_thread_tile_base_compute;

  DTYPECD *c_thread_tile_offset;
  DTYPEAB *a_thread_tile_offset_copy;
  DTYPEAB *b_thread_tile_offset_copy;
  DTYPEAB *a_thread_tile_offset_compute;
  DTYPEAB *b_thread_tile_offset_compute;
  
  // K dimension is sequential so this is not mapped to the gpu compute
  // heirarchy. Inter tile K-loop.
  for(int kk = 0; kk < k; kk += Ktile){
    //printf("kk:%d\n", kk);
    // Base address in global tile of A & B operand thread block tile.
    a_tb_tile_offset = a_tb_tile_base + i_iter_tile_base * k + kk;
    b_tb_tile_offset = b_tb_tile_base + kk * n + j_iter_tile_base;

    a_thread_tile_base_copy = a_tb_tile_offset; 
    b_thread_tile_base_copy = b_tb_tile_offset; 

    // Represents the row and col to copy by the corresponding thread in the thread block. It is not
    // the global row/col to copy, it is the row/col to copy relative to the thread block tile.
    int A_row_to_copy_in_global = linear_tid / num_threads_to_copy_one_Arow;
    int A_col_to_copy_in_global = linear_tid % num_threads_to_copy_one_Arow * Achunktocopy; 
    int B_row_to_copy_in_global = linear_tid / num_threads_to_copy_one_Brow;
    int B_col_to_copy_in_global = linear_tid % num_threads_to_copy_one_Brow * Bchunktocopy; 

    a_thread_tile_offset_copy = a_thread_tile_base_copy + A_row_to_copy_in_global * k + A_col_to_copy_in_global;
    b_thread_tile_offset_copy = b_thread_tile_base_copy + B_row_to_copy_in_global * n + B_col_to_copy_in_global; 

    // Copy the operands from global to shared memory. Each thread copies the
    // `chunktocopy` elements from global to shared memory. The thread Id's
    // inside a thread block need to be linearized. Each thread copies it's
    // contiguous chunk form global memory to the shared memroy.
    #pragma unroll
    for(int cpi = 0; cpi < Achunktocopy; ++cpi){
      asmem[linear_tid * Achunktocopy + cpi] = a_thread_tile_offset_copy[cpi];
    }

    #pragma unroll
    for(int cpi = 0; cpi < Bchunktocopy; ++cpi){
      bsmem[linear_tid * Bchunktocopy + cpi] = b_thread_tile_offset_copy[cpi]; 
    }
    __syncthreads();

    for(int i_iter_warp_base = warpIdx.y * WarpMtile; i_iter_warp_base < Mtile; i_iter_warp_base += WarpMtile * numWarpsInM){
      for(int j_iter_warp_base = warpIdx.x * WarpNtile; j_iter_warp_base < Ntile; j_iter_warp_base += WarpNtile * numWarpsInN){
	//printf("%d %d\n", i_iter_warp_base, j_iter_warp_base);
	c_warp_tile_offset = c_warp_tile_base + i_iter_warp_base * n + j_iter_warp_base;
	c_thread_tile_base = c_warp_tile_offset;

	// warp_tile_base for compute is equal to the base address in shared memory.
	a_warp_tile_base = &asmem[0]; 
	b_warp_tile_base = &bsmem[0]; 

	// Inter-warp-tile loop. Goes inside a thread block tile in steps of
	// warpKtile. Tf WarpKtile is equal to Ktile then K dimesnion is not really
	// tiled for the warp. Tiling for warp may result in reduced register pressure
	// and hence may reduce spills.
	// TODO: Try to see performance benifits of different tile sized in the k-dime
	// nsion for warps.
	for(int kkk = 0; kkk < Ktile; kkk += WarpKtile){
	  if(kkk < k){
	    //printf("kk:%d kkk:%d kk+Ktile:%d\n", kk, kkk, kk + Ktile); 

	    // Warp tile offset of asmem will only be dependent on the warpIdx.y i.e.,
	    // the row of the warp which is computing this particular part.
	    a_warp_tile_offset_compute = a_warp_tile_base + (i_iter_warp_base * Ktile) + kkk;

	    // Warp tile offset of bsmem will only be dependent on the warpIdx.x i.e.,
	    // the col of the warp which is computing this particular part.
	    b_warp_tile_offset_compute = b_warp_tile_base + (kkk * Ntile) + (j_iter_warp_base);

	    // Calculate the thread tile base and compute.
	    a_thread_tile_base_compute = a_warp_tile_offset_compute; 
	    b_thread_tile_base_compute = b_warp_tile_offset_compute; 

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
	    for(int i_iter_thread_base = warpLocalTidY; i_iter_thread_base < WarpMtile; i_iter_thread_base+=WarpdimY){
	      #pragma unroll
	      for(int j_iter_thread_base = warpLocalTidX; j_iter_thread_base < WarpNtile; j_iter_thread_base+=WarpdimX){
		a_thread_tile_offset_compute = a_thread_tile_base_compute + i_iter_thread_base * Ktile;
		b_thread_tile_offset_compute = b_thread_tile_base_compute + j_iter_thread_base;
		c_thread_tile_offset = c_thread_tile_base + i_iter_thread_base * n + j_iter_thread_base;
		// Intra-tile K-loop.
		#pragma unroll
		for(int k_iter = 0; k_iter < WarpKtile; ++k_iter){
		  // This statement now uses the shared memory fast buffers.
		  //*c_thread_tile_offset += a_thread_tile_offset_compute[k_iter] * 
		  //  b_thread_tile_offset_compute[k_iter * Ntile];
		}
	      }
	    }
	  }
	}
      }
    }
    __syncthreads();
  }
}

void hostGEMM(DTYPEAB * a, DTYPEAB * b, DTYPECD * c, int m, int n, int k){
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j ){
      DTYPECD temp = 0;
      for(int kk = 0; kk < k ; ++kk){
	//temp += a[i * k + kk] * b[kk * n + j];
      }
      c[i * n + j] = temp;
    }
  }	
}

bool compareGEMM(DTYPECD * h_c, DTYPECD * h_c_gpu_res, int m, int n){	
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j ){
      if(abs(h_c[i * n + j] - h_c_gpu_res[i * n + j]) > 1e-4)
	return false;	
    }
  }
  return true;
}

//void initMatrix(DTYPEAB * matrix, int m, int n){
//  for(int i = 0; i < m; ++i){
//    for(int j = 0; j < n; ++j){
//      matrix[i * n + j] = static_cast <DTYPEAB> (rand()) / static_cast <DTYPEAB> (RAND_MAX);
//    }
//  }
//}

void printMatrix(DTYPECD * matrix, int m, int n){
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      cout<<matrix[i * n + j]<<" ";
    }
    cout<<endl;
  }
  cout<<endl;
}

int main(){
  DTYPEAB *d_a, *d_b, *h_a, *h_b;
  DTYPECD *d_c, *h_c, *h_c_gpu_res;
  int m ,n, k;

  m = M;
  n = N;
  k = K;

  h_a = (DTYPEAB*) malloc(m * k * sizeof(DTYPEAB));
  h_b = (DTYPEAB*) malloc(k * n * sizeof(DTYPEAB));
  h_c = (DTYPECD*) malloc(m * n * sizeof(DTYPECD));
  h_c_gpu_res = (DTYPECD*) malloc(m * n * sizeof(DTYPECD));
  check_cuda_error(cudaMalloc(&d_a, m * k * sizeof(DTYPEAB)));
  check_cuda_error(cudaMalloc(&d_b, k * n * sizeof(DTYPEAB)));
  check_cuda_error(cudaMalloc(&d_c, m * n * sizeof(DTYPECD)));

  //initMatrix(h_a, m , k);	
  //initMatrix(h_b, k , n);	

  check_cuda_error(cudaMemcpy(d_a, h_a, m * k * sizeof(DTYPEAB), cudaMemcpyHostToDevice));
  check_cuda_error(cudaMemcpy(d_b, h_b, k * n * sizeof(DTYPEAB), cudaMemcpyHostToDevice));

  dim3 block(128, 1, 1);
  dim3 grid((n + Ntile - 1) / Ntile, (m + Mtile - 1) / Mtile, 1);

  //printf("%d %d %d\n", block.x, block.y, block.z);
  //printf("%d %d %d\n", grid.x, grid.y, grid.z);
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
  cudaMemcpy(h_c_gpu_res, d_c, m * n * sizeof(DTYPECD), cudaMemcpyDeviceToHost);
  hostGEMM(h_a, h_b, h_c, m, n, k);

  if(compareGEMM(h_c, h_c_gpu_res, m, n))
    cout<<"Success!\n";
  else
    cout<<"Output does not match!\n";

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
