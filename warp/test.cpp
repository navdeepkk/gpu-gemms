//  nvcc -O3 -std=c++11 -use_fast_math -ccbin g++ -arch=compute_75 -code=sm_75 -expt-relaxed-constexpr 
//  Performs matrix mutliplication using shared memory tiles where ewach thread
//  may need to calculate and move more than one data element. Assumes matrices
//  stored in row major order. The loop structure followed is as(one level
//  tiling)
//  
//  for(int i = 0; i < M; i += Mtile){			      //Inter-tb-Tile
//    for(int j = 0; j < N; j += Ntile){		      //Inter-tb-Tile
//	for(int k = 0; k < K; k += Ktile){		      //Inter-tb-Tile
//	  for(int ii = i; ii < i + Mtile; ++ii){	      //Intra-tb-Tile
//	    for(int jj = j; jj < j + Ntile; ++jj){	      //Intra-tb-Tile
//	      for(int iii = ii; iii < ii + Mchunk; ++iii){    //Per-Thread
//		for(int jjj = jj; jjj < jj + Nchunk; ++jjj){  //Per-Thread
//		  for(int kk = k; kk < k + Ktile; ++kk){      //Intra-tb-Tile
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
#define M 128 
#define N 128 
#define K 128 
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
// have 32 x 4 = 128 threads. A thread block of the kind 32 x 4 etc. is not possible,
// because weneed to have the thread block dimensions as multiples of 32 which 4 clearly
// isn't. so we need to launch 128 threads in one dimension only a whole lot of refactoring 
// is needed.

using namespace std;

typedef struct {
  unsigned x;
  unsigned y;
  unsigned z;
} WarpIdx;

__global__ void GEMM(DTYPE * a, DTYPE * b, DTYPE * c, int m, int n, int k){
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

  // Reserve shared memory tiles if to put in the operands.
  __shared__ DTYPE asmem[Mtile * Ktile]; 
  __shared__ DTYPE bsmem[Ktile * Ntile];

  // Since the actual computation tile size is greater than than the thread
  // block tile size, therefore we want to find out what size of the output tile
  // is a thread calculating.
  // Now each thread will compute an output tile of size (Mchunk, Nchunk).
  constexpr int Mchunk = WarpMtile / WarpdimY;
  constexpr int Nchunk = WarpNtile / WarpdimX;

  // Calculate the chunk of data each thread has to copy from the global memroy
  // to shared memeory. It is equal to the total number of data elements in a
  // Tile / total number of threads in a thread block.
  // TODO: Insert checks here to see if the if the tile size in elements is less than
  // the number of threads in a thread block.
  int Achunktocopy = (Mtile * Ktile) / numThreads;
  int Bchunktocopy = (Ktile * Ntile) / numThreads;

  // Linear thread id in the thread block.
  int linear_tid = (threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

  // Linear warp id in the thread block.
  int linear_warpid = linear_tid / WarpSize;

  // Since warps are first linearized in the `x` dimesnions then `y` then `z`.
  warpIdx.x = linear_warpid % numWarpsInN;
  warpIdx.y = linear_warpid / numWarpsInN;
  warpIdx.z = 1;

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
  // All the indexes defined below are local the chunk that identity is calculating.
  // for e.g., i_iter_warp_base maps to the local `i` iteration in the tile that
  // this warp is calculating.
  int i_iter_tile_base = blockIdx.y * Mtile; // Maps to inter-tile `i`.
  int j_iter_tile_base = blockIdx.x * Ntile; // Maps to inter-tile `j`.

  // The warp iteration that is being carried out.
  //int i_iter_warp_base = warpIdx.y * WarpMtile; // Maps to inter-warp-tile `i`.
  //int j_iter_warp_base = warpIdx.x * WarpNtile; // Maps to inter-warp-tile `j`.

  int warpLocalTidY = (linear_tid % WarpSize) / WarpdimX;
  int warpLocalTidX = (linear_tid % WarpSize) % WarpdimX;

  // The Inner loop iteration beginning that this thread is carruing out.
  int i_iter_thread_base = warpLocalTidY * Mchunk; // Maps to thread iteration.
  int j_iter_thread_base = warpLocalTidX * Nchunk; // Maps to thread iteration.
  //printf("%d %d \n", i_iter_thread_base, j_iter_thread_base);

  // Number of threads required to copy one row of A from global to shared memory.
  int num_threads_to_copy_one_Arow = Ktile / Achunktocopy;
  // Number of threads required to copy one row of B form global to shared memory.
  int num_threads_to_copy_one_Brow = Ntile / Bchunktocopy;

  DTYPE *c_tb_tile_base = c;
  DTYPE *a_tb_tile_base = a;
  DTYPE *b_tb_tile_base = b;

  DTYPE *c_tb_tile_offset = c_tb_tile_base + i_iter_tile_base * n + j_iter_tile_base;
  DTYPE *a_tb_tile_offset;
  DTYPE *b_tb_tile_offset;

  DTYPE *c_warp_tile_base = c_tb_tile_offset;
  DTYPE *a_warp_tile_base;
  DTYPE *b_warp_tile_base;

  // c_warp_tile_offset will be in the global memory tile
  // while for A and B they will be in the sharem memory.
  DTYPE *c_warp_tile_offset;
  DTYPE *a_warp_tile_offset_compute;
  DTYPE *b_warp_tile_offset_compute;

  DTYPE *c_thread_tile_base;
  DTYPE *a_thread_tile_base_copy;
  DTYPE *b_thread_tile_base_copy;
  DTYPE *a_thread_tile_base_compute;
  DTYPE *b_thread_tile_base_compute;

  DTYPE *c_thread_tile_offset;
  DTYPE *a_thread_tile_offset_copy;
  DTYPE *b_thread_tile_offset_copy;
  DTYPE *a_thread_tile_offset_compute;
  DTYPE *b_thread_tile_offset_compute;

  // Allocate a Ctile in registers of dimensions (Mchunk, Nchunk).
  DTYPE Cout[Mchunk * Nchunk];
  for(int i = 0; i < Mchunk; ++i){
    for(int j = 0; j < Nchunk; ++j){
      Cout[i * Nchunk + j] = 0.0f;
    }
  }

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
	//if(i_iter_warp_base == 64 && j_iter_warp_base == 0)
	//  printf("%d %d\n", i_iter_warp_base, j_iter_warp_base);
	c_warp_tile_offset = c_warp_tile_base + i_iter_warp_base * n + j_iter_warp_base;
	c_thread_tile_base = c_warp_tile_offset;
	c_thread_tile_offset = c_thread_tile_base + i_iter_thread_base * n + j_iter_thread_base;

	// warp_tile_base for compute is equal to the base address in shared memory.
	a_warp_tile_base = &asmem[0];
	b_warp_tile_base = &bsmem[0];
	
	// Inter-warp-tile loop. Goes inside a thread block tile in steps of
	// warpKtile. If WarpKtile is equal to Ktile then K dimesnion is not really
	// tiled for the warp. Tiling for warp may result in reduced register pressure
	// and hence may reduce spills.
	// TODO: Try to see performance benifits of different tile sized in the k-dime
	// nsion for warps.
	for(int kkk = kk; kkk < kk + Ktile; kkk += WarpKtile){
	  if(kkk < k){
	    //printf("kk:%d kkk:%d kk+Ktile:%d\n", kk, kkk, kk + Ktile);
	    // Warp tile offset of asmem will only be dependent on the warpIdx.y i.e.,
	    // the row of the warp which is computing this particular part.
	    a_warp_tile_offset_compute = a_warp_tile_base + (i_iter_warp_base * Ktile) + kkk;

	    // Warp tile offset of bsmem will only be dependent on the warpIdx.x i.e.,
	    // the col of the warp which is computing this particular part.
	    b_warp_tile_offset_compute = b_warp_tile_base + (kkk * Ntile) + (j_iter_warp_base);

	    a_thread_tile_base_compute = a_warp_tile_offset_compute;
	    b_thread_tile_base_compute = b_warp_tile_offset_compute;

	    a_thread_tile_offset_compute = a_thread_tile_base_compute + i_iter_thread_base * Ktile;
	    b_thread_tile_offset_compute = b_thread_tile_base_compute + j_iter_thread_base;

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
	    for(int i_iter = 0, ci = 0; i_iter < Mchunk; ++i_iter, ++ci){
#pragma unroll
	      for(int j_iter = 0, cj = 0; j_iter < Nchunk; ++j_iter, ++cj){
		//if(b_thread_tile_offset_compute > &bsmem[0] + Ntile * Ktile)
		//  printf("xx\n");
		// Intra-tile K-loop.
#pragma unroll
		for(int k_iter = 0; k_iter < WarpKtile; ++k_iter){
		  //printf("i:%d, j:%d, k:%d\n", i_iter, j_iter, k_iter);
		  if(i_iter < Mtile && j_iter < Ntile){
		    // This statement now uses the shared memory fast buffers.
		    Cout[ci * Nchunk + cj] += 1;//a_thread_tile_offset_compute[i_iter * Ktile + k_iter] * 1;//   b_thread_tile_offset_compute[k_iter * Ntile + j_iter];
		  }
		}
	      }
	    }

	    // Write back the result to the output matrix.
	    for(int ii = 0; ii < Mchunk; ++ii){
	      for(int jj = 0; jj < Nchunk; ++jj){
		//printf("%f\n", Cout[ii * Nchunk + jj]);
		c_thread_tile_offset[ii * n + jj] = Cout[ii * Nchunk + jj];
	      }
	    }
	  }
	}
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
      matrix[i * n + j] = 1;//static_cast <DTYPE> (rand()) / static_cast <DTYPE> (RAND_MAX);
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

  dim3 block(64, 1, 1);
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
  cudaMemcpy(h_c_gpu_res, d_c, m * n * sizeof(DTYPE), cudaMemcpyDeviceToHost);
  //hostGEMM(h_a, h_b, h_c, m, n, k);

  //if(compareGEMM(h_c, h_c_gpu_res, m, n))
  //  cout<<"Success!\n";
  //else
  //  cout<<"Output does not match!\n";

  //printMatrix(h_c, m, n);
  //cout<<"output gpu\n";
  printMatrix(h_c_gpu_res, m, n);

  free(h_a);
  free(h_b);
  free(h_c);
  free(h_c_gpu_res);

  // The Global index start that this thread is responsible for.cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
