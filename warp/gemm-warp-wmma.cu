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
//		      //mma.sync()
//		    }
//		  }
//		}
//	      }
//	    }
//	  }
//	}
//    }
//  }

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <assert.h>
#include"common.h"

#define DTYPECD float
#define DTYPEAB __half
#define M 1024 
#define N 1024 
#define K 1024 
#define WM 16
#define WN 16
#define WK 16
#define Mtile 64  // This will actually be the loop step of `i` loop.
#define Ntile 64  // This will actually be the loop step of `j` loop.
#define Ktile 32  // This will actually be the loop step of `k` loop.
#define WarpMtile 32
#define WarpNtile 32
#define WarpKtile 16 // 16 because the size supported by the wmma api is 16x16x16.
#define WarpSize 32

#define C_LAYOUT wmma::mem_row_major

using namespace nvcuda;

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

__host__ void init_host_matrices(DTYPEAB *a, DTYPEAB *b, DTYPECD *c){
  for(int i = 0; i < M; i++){
    for(int j = 0; j < K; j++){
      a[i * K + j] = (half) (rand() / RAND_MAX);//1.0f;
    }
  }

  for(int i = 0; i < K; i++){
    for (int j = 0; j < N; j++){
      b[i * N + j] = (half) (rand() / RAND_MAX);//1.0f;
    }
  }

  for (int t = 0; t < M * N; t++) {
    c[t] = (DTYPECD) 0.0f;
  }
}

template<typename T>
__host__ void printMatrix(T * matrix, int m, int n){
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      printf("%f ", (matrix[i * n + j]));
    }
    printf("\n");
  }
  printf("\n");
}

__global__ void GEMM(DTYPEAB * a, DTYPEAB * b, DTYPECD * c, DTYPECD * d, int m, int n, int k){
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
  __shared__ DTYPECD csmem[Mtile * Ntile];

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

  DTYPECD *c_tb_tile_base = c;
  DTYPECD *d_tb_tile_base = d;
  DTYPEAB *a_tb_tile_base = a;
  DTYPEAB *b_tb_tile_base = b;

  DTYPECD *c_tb_tile_offset = c_tb_tile_base + i_iter_tile_base * n + j_iter_tile_base;
  DTYPECD *d_tb_tile_offset = d_tb_tile_base + i_iter_tile_base * n + j_iter_tile_base;
  DTYPEAB *a_thread_tile_base_copy;
  DTYPEAB *b_thread_tile_base_copy;
  DTYPECD *c_thread_tile_base_copy = c_tb_tile_offset;
  DTYPEAB *a_tb_tile_offset;
  DTYPEAB *b_tb_tile_offset;

  DTYPECD *c_warp_tile_base;
  DTYPECD *d_warp_tile_base;
  DTYPEAB *a_warp_tile_base;
  DTYPEAB *b_warp_tile_base;

  // c_warp_tile_offset will be in the global memory tile
  // while for A and B they will be in the shared memory.
  DTYPECD *c_warp_tile_offset;
  DTYPECD *d_warp_tile_offset;
  DTYPEAB *a_warp_tile_offset_compute;
  DTYPEAB *b_warp_tile_offset_compute;

  // warp_tile_base for compute is equal to the base address in shared memory.
  a_warp_tile_base = &asmem[0];
  b_warp_tile_base = &bsmem[0];
  c_warp_tile_base = &csmem[0];
  d_warp_tile_base = d_tb_tile_offset;

  // Allocate accmulator fragments for the C warp tiles. The allocation happens at a per
  // warp level. Each warp in the thread block will have this type of accumulator tile.
  // This accumulator tile is to be kept alive even accross different iterations of the
  // outermost k-loop. The 
  wmma::fragment<wmma::accumulator, WM, WN, WK, DTYPECD> c_accum[WarpMtile / WM][WarpNtile / WN];
  
  // Copy the c matrix into the shared memory memory for scaling.
  #pragma unroll
  for(int i = linear_tid, e = Mtile * Ntile, x = blockDim.x * blockDim.y; i < e; i+= x){
    csmem[((i / Ntile) * Ntile) + (i % Ntile)] = c_thread_tile_base_copy[((i / Ntile) * n) + (i % Ntile)];
  }
  __syncthreads();
  
  //------Write code for fractional scaling here----//
  //
  //
  //------------------------------------------------//

  // K dimension is sequential so this is not mapped to the gpu compute
  // heirarchy. Inter tile K-loop.
  for(int kk = 0; kk < k; kk += Ktile){
    //printf("kk:%d\n", kk);
    // Base address in global tile of A & B operand thread block tile.
    a_tb_tile_offset = a_tb_tile_base + i_iter_tile_base * k + kk;
    b_tb_tile_offset = b_tb_tile_base + kk * n + j_iter_tile_base;

    a_thread_tile_base_copy = a_tb_tile_offset;
    b_thread_tile_base_copy = b_tb_tile_offset;
    // Copy the operands from global to shared memory. Each thread copies the
    // `chunktocopy` elements from global to shared memory. The thread Id's
    // inside a thread block need to be linearized. Each thread copies it's
    // contiguous chunk form global memory to the shared memroy.
    #pragma unroll
    for(int i = linear_tid, e = Mtile * Ktile, x = blockDim.x * blockDim.y; i < e; i+= x){
      asmem[((i / Ktile) * Ktile) + (i % Ktile)] = a_thread_tile_base_copy[((i / Ktile) * k) + (i % Ktile)];
    }

    #pragma unroll
    for(int i = linear_tid, e = Ntile * Ktile, x = blockDim.x * blockDim.y; i < e; i+= x){
      bsmem[((i / Ktile) * Ktile) + (i % Ktile)] = b_thread_tile_base_copy[((i / Ntile) * n) + (i % Ntile)];
    } 
    __syncthreads();

    // These loops goes over warp tiles of dimension (WarpMtile, WarpNtile) inside the thread block tile. 
    // Useful when the number of warp tiles is more than the numver of warps available. I.e., one warp
    // is responsible for more than one warp tile.
    for(int i_iter_warp_base = warpIdx.y * WarpMtile; i_iter_warp_base < Mtile; i_iter_warp_base += WarpMtile * numWarpsInM){
      for(int j_iter_warp_base = warpIdx.x * WarpNtile; j_iter_warp_base < Ntile; j_iter_warp_base += WarpNtile * numWarpsInN){
        //printf("%d %d\n", i_iter_warp_base, j_iter_warp_base).	
        c_warp_tile_offset = c_warp_tile_base + i_iter_warp_base * Ntile + j_iter_warp_base;
        d_warp_tile_offset = d_warp_tile_base + i_iter_warp_base * n + j_iter_warp_base;

        // The wmma api only supports accumulator frags of size 16x16.
        // We need to copy the corresponding tiles from shared memory to
        // the registers in chunks of 16x16 each. This loop is meant to run
	// only once per warp tile.
	if(kk == 0){
	  for(int i = 0; i < WarpMtile; i += WM){
            for(int j = 0; j < WarpNtile; j += WN){
              wmma::load_matrix_sync(c_accum[i / WM][j / WN], c_warp_tile_offset +
	  					(i * Ntile) + j, Ntile, C_LAYOUT);
            }
	  }
	}
        
        // Inter-warp-tile loop. Goes inside a thread block tile in steps of
        // warpKtile. Tf WarpKtile is equal to Ktile then K dimesnion is not really
        // tiled for the warp. Tiling for warp may result in reduced register pressure
        // and hence may reduce spills.
        // TODO: Try to see performance benifits of different tile sized in the k-dime
        // nsion for warps.
        // TODO: This needs to be fixed. If the WarpKtile is not 16 then the number of
        // 16x16 operand tiles that are moved need to be changed. I.e., a_frag and b_frag
        // now need to be 2-d arrays and one more loop inside this loop needs to be present
        // which calculates the WarpKtile in chunks of 16 each.
	#pragma unroll
        for(int kkk = 0; kkk < Ktile; kkk += WarpKtile){
          if(kkk < k){
            //printf("kk:%d kkk:%d kk+Ktile:%d\n", kk, kkk, kk + Ktile);
           
            // These fragments contain the register operands for the a and b matrix. These
            // contain only the operands for calculating one k-dimension.
            wmma::fragment<wmma::matrix_a, WM, WN, WK, DTYPEAB, wmma::row_major> a_frag[WarpMtile / WM];
            wmma::fragment<wmma::matrix_b, WM, WN, WK, DTYPEAB, wmma::row_major> b_frag[WarpNtile / WN];
            
            // Warp tile offset of asmem will only be dependent on the warpIdx.y i.e.,
            // the row of the warp which is computing this particular part. This points to the
            // starting address of this warp for the `a` operand.
            a_warp_tile_offset_compute = a_warp_tile_base + (i_iter_warp_base * Ktile) + kkk;

            // Warp tile offset of bsmem will only be dependent on the warpIdx.x i.e.,
            // the col of the warp which is computing this particular part. This points to the
            // starting address of this warp for the `b` operand.
            b_warp_tile_offset_compute = b_warp_tile_base + (kkk * Ntile) + (j_iter_warp_base);

            // Compute the warp tile in chunks of (WM, WN). Move the fragments into the registers on the go.
	    #pragma unroll
            for(int i = 0; i < WarpMtile; i += WM){
              wmma::load_matrix_sync(a_frag[i], a_warp_tile_offset_compute + (i * Ktile), Ktile);
	      #pragma unroll
              for(int j = 0; j < WarpNtile; j += WN){
                if(i == 0){
                  // copy the bfragments only once.
                  wmma::load_matrix_sync(b_frag[j], b_warp_tile_offset_compute + j, Ntile);
                }
                // call mma.sync();
                wmma::mma_sync(c_accum[i/ WM][j / WN], a_frag[i], b_frag[j], c_accum[i / WM][j / WN]);
              }
            }
          }
        }
      }
    }
  }
  
  //DTYPECD* cd = d_tb_tile_base + 1;
  // K-dimension processing of one warp is finished. We can copy the accum fragment
  // corresponding to this warp to the result array `d` in global memory.
  // TODO: currently assuming that one warp tile is only mapped to one warp, i.e.,
  // one warp needs to calculate only one warp tile. Hence d_warp_tile_offset is known
  // form the last time it was calculated and hence reused here.
  for(int i = 0; i < WarpMtile; i += WM){
    for(int j = 0; j < WarpNtile; j += WN){
      wmma::store_matrix_sync(d_warp_tile_offset + ((i * n) + j), c_accum[i / WM][j/ WN], n, C_LAYOUT);
    }
  }
}

void hostGEMM(DTYPEAB * a, DTYPEAB * b, DTYPECD * c, int m, int n, int k){
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j ){
      DTYPECD temp = 0;
      for(int kk = 0; kk < k ; ++kk){
	temp += (float)a[i * k + kk] * (float)b[kk * n + j];
      }
      c[i * n + j] = temp;
    }
  }
}

bool compareGEMM(DTYPECD * h_c, DTYPECD * h_c_gpu_res, int m, int n){
  for (int i = 0; i < N * M; i++) {
    if(fabs(h_c_gpu_res[i] - h_c[i]) > 0.1f){
      printf("mismatch i=%d result_hD=%f result_host=%f\n", i, h_c_gpu_res[i], h_c[i]);
      return false;
    }
  }
  return true;
}

int main(){
  DTYPEAB *d_a, *d_b, *h_a, *h_b;
  DTYPECD *d_c, *d_d, *h_c, *h_c_gpu_res;
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
  check_cuda_error(cudaMalloc(&d_d, m * n * sizeof(DTYPECD)));

  assert(((unsigned long long)d_a) % 128 == 0);
  assert(((unsigned long long)d_b) % 128 == 0);
  assert(((unsigned long long)d_c) % 128 == 0);
  assert(((unsigned long long)d_d) % 128 == 0);

  init_host_matrices(h_a, h_b, h_c);

  check_cuda_error(cudaMemcpy(d_a, h_a, m * k * sizeof(DTYPEAB), cudaMemcpyHostToDevice));
  check_cuda_error(cudaMemcpy(d_b, h_b, k * n * sizeof(DTYPEAB), cudaMemcpyHostToDevice));
  check_cuda_error(cudaMemcpy(d_c, h_c, m * n * sizeof(DTYPECD), cudaMemcpyHostToDevice));

  dim3 block(128, 1, 1);
  dim3 grid((n + Ntile - 1) / Ntile, (m + Mtile - 1) / Mtile, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  GEMM<<<grid, block>>>(d_a, d_b, d_c, d_d, m , n, k);
  cudaEventRecord(stop, NULL);

  cudaEventSynchronize(stop);
  float msecTotal = 0.0f;
  cudaEventElapsedTime(&msecTotal, start, stop);
  double flopsPerMatrixMul = 2.0 * (double) m * (double) n * (double) k;
  double teraFlops = (flopsPerMatrixMul * 1.0e-12f) / (msecTotal / 1000.0f);
  cout<<"PERF: "<<teraFlops<<"Tflops \n";

  check_cuda_error(cudaPeekAtLastError());
  check_cuda_error(cudaDeviceSynchronize());
  cudaMemcpy(h_c_gpu_res, d_d, m * n * sizeof(DTYPECD), cudaMemcpyDeviceToHost);
  
  #ifdef DEBUG
  hostGEMM(h_a, h_b, h_c, m, n, k);

  if(compareGEMM(h_c, h_c_gpu_res, m, n))
    cout<<"Success!\n";
  else
    cout<<"Output does not match!\n";
  #endif
  
  #ifdef PRINT_HOST
  printMatrix(h_c, m, n);
  #endif
  
  #ifdef PRINT_GPU
  printMatrix(h_c_gpu_res, m, n);
  #endif
  
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_c_gpu_res);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_d);

  return 0;
}
