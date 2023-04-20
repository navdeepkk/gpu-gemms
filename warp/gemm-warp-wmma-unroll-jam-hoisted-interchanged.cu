// nvcc -Xptxas -O3 -maxrregcount=255 -std=c++11 -use_fast_math -ccbin g++
// -arch=compute_86 -code=sm_86 -DDEBUG_GPU
//
//  Performs matrix multiplication using shared memory tiles where each thread
//  may need to calculate and move more than one data element. Assumes matrices
//  stored in row major order. The loop structure followed is as(one level
//  tiling)

#include <assert.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>

#include <cmath>
#include <cuda/pipeline>

#include "common.h"
#include "include/gemm_params.h"

#define DTYPECD float
#define DTYPEAB __half
#define M 1024
#define N 3072
#define K 768
#define WarpSize 32
#define NUM_THREADS_PER_BLOCK \
  (Mtile / WarpMtile) * (Ntile / WarpNtile) * WarpSize
#define MBLOCK 32
#define NBLOCK 32

#define C_LAYOUT wmma::mem_row_major

using namespace nvcuda;

using namespace std;

typedef struct {
  unsigned x;
  unsigned y;
  unsigned z;
} WarpIdx;

__host__ void init_host_matrices(DTYPEAB *a, DTYPEAB *b, DTYPECD *c) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      a[i * K + j] = __float2half(static_cast<float>(
          static_cast<int>(static_cast<float>(rand()) /
                           static_cast<float>(RAND_MAX) * 100) +
          1));
    }
  }

  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      b[i * N + j] = __float2half(static_cast<float>(
          static_cast<int>(static_cast<float>(rand()) /
                           static_cast<float>(RAND_MAX) * 100) +
          1));
    }
  }

  for (int t = 0; t < M * N; t++) {
    c[t] = (DTYPECD)0.0f;
  }
}

__host__ void printMatrix(DTYPEAB *matrix, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%f ", __half2float((matrix[i * n + j])));
    }
    printf("\n");
  }
  printf("\n");
}

__host__ void printMatrixFloat(DTYPECD *matrix, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%f ", matrix[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

__global__ void GEMM_NAIVE(DTYPEAB *a, DTYPEAB *b, DTYPECD *c, int m, int n,
                           int k) {
  int i_iter = blockIdx.y * blockDim.y + threadIdx.y;
  int j_iter = blockIdx.x * blockDim.x + threadIdx.x;

  DTYPECD temp = 0.0f;

  for (int kk = 0; kk < k; ++kk) {
    if (i_iter < m && j_iter < n) {
      temp +=
          __half2float(a[i_iter * k + kk]) * __half2float(b[kk * n + j_iter]);
    }
  }

  c[i_iter * n + j_iter] = temp;
}

__global__ void GEMM(DTYPEAB *a, DTYPEAB *b, DTYPECD *c, DTYPECD *d, int m,
                     int n, int k) {
  // Struct holding the geometrical coordinates of the warp.
  WarpIdx warpIdx;

  constexpr int numThreads = NUM_THREADS_PER_BLOCK;
  int numWarpsInM = 1, numWarpsInN = 1, numWarps = numThreads / WarpSize;

  // Number of warps in the `M` and `N` dimension, of the thread block. If there
  // are sufficient number of warps for the `N` dimension then use them. If not
  // assign all of them to the `N` dimension. If excessive are present then
  // assign the remaining them to the `M` dimension.
  if (numWarps <= Ntile / WarpNtile) {
    numWarpsInN = numWarps;
  } else {
    numWarpsInN = Ntile / WarpNtile;
    numWarpsInM = numWarps / numWarpsInN;
  }

  constexpr int smem_stage_offset = (STAGES == 1 ? 1 : ((STAGES - 1) * 2));

  // Reserve shared memory tiles for the operands.
  extern __shared__ int s[];
  DTYPEAB *asmem = (DTYPEAB *)s;
  DTYPEAB *bsmem = &asmem[smem_stage_offset * Mtile * (Ktile + PADDING_A)];

  // Linear thread id in the thread block.
  int linear_tid = (threadIdx.z * blockDim.x * blockDim.y) +
                   (threadIdx.y * blockDim.x) + threadIdx.x;

  // i Linear warp id in the thread block.
  int linear_warpid = linear_tid / WarpSize;

  warpIdx.x = linear_warpid % numWarpsInN;
  warpIdx.y = linear_warpid / numWarpsInN;
  warpIdx.z = 1;

  // Find the iteration of the original loop nest that maps to this thread
  // block here.
  // It is more elegant to map the iterations instead of row or col. At the end
  // it doesn't matter because the iterations actually determine which row or
  // col is it.
  // The Outer loops iteration beginning that this thread block tile
  // is responsible for. These coordinates also marks the beginning of the
  // address a thread block needs to copy form the global memory to shared
  // memory. This represents the coordinates in the data space not in the GPU
  // (processor) space.
  int i_iter_tile_base =
      blockIdx.y * Mtile;  // Maps to inter-tile `i`. Global row
  int j_iter_tile_base =
      blockIdx.x * Ntile;  // Maps to inter-tile `j`. Global row

  DTYPECD *c_tb_tile_base = c;
  DTYPECD *d_tb_tile_base = d;
  DTYPEAB *a_tb_tile_base = a;
  DTYPEAB *b_tb_tile_base = b;

  DTYPECD *c_tb_tile_offset =
      c_tb_tile_base + i_iter_tile_base * n + j_iter_tile_base;
  DTYPECD *d_tb_tile_offset =
      d_tb_tile_base + i_iter_tile_base * n + j_iter_tile_base;
  DTYPEAB *a_thread_tile_base_copy;
  DTYPEAB *b_thread_tile_base_copy;
  DTYPEAB *a_tb_tile_offset;
  DTYPEAB *b_tb_tile_offset;

  DTYPECD *c_warp_tile_base;
  DTYPECD *d_warp_tile_base;
  DTYPEAB *a_warp_tile_base;
  DTYPEAB *b_warp_tile_base;

  DTYPECD *c_warp_tile_offset;
  DTYPECD *d_warp_tile_offset;
  DTYPEAB *a_warp_tile_offset_compute;
  DTYPEAB *b_warp_tile_offset_compute;

  // We try to do the address calculation in the global memory with as
  // less address computation as possible. Hence, we compute the base
  // for each threads copy here and then just add some minimal offset
  // later.
  int a_global_row = linear_tid / (Ktile / 8);
  int a_global_col = linear_tid % (Ktile / 8);
  constexpr int a_row_span_in_thread_block = numThreads / (Ktile / 8);
  constexpr int a_offset_per_thread_gmem = a_row_span_in_thread_block * (K / 8);
  int a_smem_row = a_global_row;
  int a_smem_col = a_global_col;
  constexpr int a_row_span_in_smem = a_row_span_in_thread_block;
  constexpr int a_offset_per_thread_smem =
      a_row_span_in_smem * ((Ktile + PADDING_A) / 8);
  int b_global_row = linear_tid / (Ntile / 8);
  int b_global_col = linear_tid % (Ntile / 8);
  constexpr int b_row_span_in_thread_block = numThreads / (Ntile / 8);
  constexpr int b_offset_per_thread_gmem = b_row_span_in_thread_block * (N / 8);
  int b_smem_row = b_global_row;
  int b_smem_col = b_global_col;
  constexpr int b_row_span_in_smem = b_row_span_in_thread_block;
  constexpr int b_offset_per_thread_smem =
      b_row_span_in_smem * ((Ntile + PADDING_B) / 8);

  // warp_tile_base for compute is equal to the base address in shared memory.
  c_warp_tile_base = c_tb_tile_offset;
  d_warp_tile_base = d_tb_tile_offset;

  // Allocate accumulator fragments for the C warp tiles. The allocation
  // happens at a per warp level. Each warp in the thread block will have this
  // type of accumulator tile. This accumulator tile is to be kept alive even
  // across different iterations of the outermost k-loop.
  wmma::fragment<wmma::accumulator, WM, WN, WK, DTYPECD>
      c_accum[WarpMtile / WM][WarpNtile / WN];

  // Create the pipeline object and a vector type.
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape4 = cuda::aligned_size_t<alignof(int4)>(sizeof(int4));

  // Load the accumulator tile from global memory to shared memory.
  int4 *cgmemBase = (int4 *)c_tb_tile_offset;
  int4 *csmemBase = (int4 *)s;
#pragma unroll
  for (int i = 0, e = Mtile * (Ntile / 4); i < e; i += numThreads) {
    pipe.producer_acquire();
    cuda::memcpy_async(
        (csmemBase +
         (((i + linear_tid) / (Ntile / 4)) * ((Ntile + PADDING_C) / 4)) +
         ((i + linear_tid) % (Ntile / 4))),
        (cgmemBase + (((i + linear_tid) / (Ntile / 4) * (N / 4)) +
                      ((i + linear_tid) % (Ntile / 4)))),
        shape4, pipe);
    pipe.producer_commit();
  }
  cuda::pipeline_consumer_wait_prior<0>(pipe);
  __syncthreads();

  // Load the accumulator tile from shared memory to the registers.
  c_warp_tile_base = (DTYPECD *)s;
#pragma unroll
  for (int i_iter_warp_base = warpIdx.y * WarpMtile; i_iter_warp_base < Mtile;
       i_iter_warp_base += WarpMtile * numWarpsInM) {
#pragma unroll
    for (int j_iter_warp_base = warpIdx.x * WarpNtile; j_iter_warp_base < Ntile;
         j_iter_warp_base += WarpNtile * numWarpsInN) {
      c_warp_tile_offset = c_warp_tile_base +
                           i_iter_warp_base * (Ntile + PADDING_C) +
                           j_iter_warp_base;
#pragma unroll
      for (int i = 0; i < WarpMtile; i += WM) {
#pragma unroll
        for (int j = 0; j < WarpNtile; j += WN) {
          wmma::load_matrix_sync(
              c_accum[i / WM][j / WN],
              c_warp_tile_offset + ((i * (Ntile + PADDING_C)) + j),
              (Ntile + PADDING_C), C_LAYOUT);
        }
      }
    }
  }

  // Fetch the pipelined stages AOT.
#pragma unroll
  for (int stage = 0; stage < ((STAGES - 1) * Ktile); stage += Ktile) {
    // Base address in global tile of A & B operand thread block tile.
    a_tb_tile_offset = a_tb_tile_base + i_iter_tile_base * k + stage;
    b_tb_tile_offset = b_tb_tile_base + stage * n + j_iter_tile_base;

    a_thread_tile_base_copy = a_tb_tile_offset;
    b_thread_tile_base_copy = b_tb_tile_offset;
    int4 *agmemBase =
        (int4 *)a_thread_tile_base_copy + a_global_row * (K / 8) + a_global_col;
    int4 *asmemBase =
        (int4 *)&asmem[(stage / Ktile) * (Mtile * (Ktile + PADDING_A))];
    asmemBase = asmemBase + a_smem_row * ((Ktile + PADDING_A) / 8) + a_smem_col;
    int4 *bgmemBase =
        (int4 *)b_thread_tile_base_copy + b_global_row * (N / 8) + b_global_col;
    int4 *bsmemBase =
        (int4 *)&bsmem[(stage / Ktile) * (Ktile * (Ntile + PADDING_B))];
    bsmemBase = bsmemBase + b_smem_row * ((Ntile + PADDING_B) / 8) + b_smem_col;

    pipe.producer_acquire();
#pragma unroll
    for (int i = 0, e = (Mtile * (Ktile / 8)) / numThreads; i < e; ++i) {
      cuda::memcpy_async(asmemBase, agmemBase, shape4, pipe);
      asmemBase = asmemBase + a_offset_per_thread_smem;
      agmemBase = agmemBase + a_offset_per_thread_gmem;
    }
#pragma unroll
    for (int i = 0, e = (Ktile * (Ntile / 8)) / numThreads; i < e; ++i) {
      cuda::memcpy_async(bsmemBase, bgmemBase, shape4, pipe);
      bsmemBase = bsmemBase + b_offset_per_thread_smem;
      bgmemBase = bgmemBase + b_offset_per_thread_gmem;
    }
    pipe.producer_commit();
  }

// These loops goes over warp tiles of dimension (WarpMtile, WarpNtile) inside
// the thread block tile. Useful when the number of warp tiles is more than
// the number of warps available. I.e., one warp is responsible for more than
// one warp tile. Ideally this loop will both these loops must have a single
// iteration.
#pragma unroll
  for (int i_iter_warp_base = warpIdx.y * WarpMtile; i_iter_warp_base < Mtile;
       i_iter_warp_base += WarpMtile * numWarpsInM) {
#pragma unroll
    for (int j_iter_warp_base = warpIdx.x * WarpNtile; j_iter_warp_base < Ntile;
         j_iter_warp_base += WarpNtile * numWarpsInN) {
      d_warp_tile_offset =
          d_warp_tile_base + i_iter_warp_base * N + j_iter_warp_base;

      // Inter-warp-tile loop. Goes inside a thread block tile in steps of
      // warpKtile. Tf WarpKtile is equal to Ktile then K dimension is not
      // really tiled for the warp. Tiling for warp may result in reduced
      // register pressure and hence may reduce spills.
      // TODO: Try to see performance benefits of different tile sized in the
      // k-dimension for warps.
      // TODO: This needs to be fixed. If the WarpKtile is not 16 then the
      // number of 16x16 operand tiles that are moved need to be changed. I.e.,
      // a_frag and b_frag now need to be 2-d arrays and one more loop inside
      // this loop needs to be present which calculates the WarpKtile in chunks
      // of 16 each.

      // These fragments contain the register operands for the A and B matrix.
      // These contain only the operands for calculating one k-dimension.
      wmma::fragment<wmma::matrix_a, WM, WN, WK, DTYPEAB, wmma::row_major>
          a_frag[WarpMtile / WM];
      wmma::fragment<wmma::matrix_b, WM, WN, WK, DTYPEAB, wmma::row_major>
          b_frag;

      // K dimension is sequential so this is not mapped to the gpu compute
      // hierarchy. Inter tile K-loop. Thread Block K-loop.
#pragma unroll
      for (int kk = (STAGES - 1) * Ktile; kk < k; kk += Ktile * (STAGES - 1)) {
        // Fetch ahead for `STAGES - 1` stages.
#pragma unroll
        for (int stage = kk;
             stage < kk + (STAGES == 1 ? 1 : ((STAGES - 1)) * Ktile) &&
             stage < k;
             stage += Ktile) {
          // Base address in global tile of A & B operand thread block tile.
          a_tb_tile_offset = a_tb_tile_base + i_iter_tile_base * k + stage;
          b_tb_tile_offset = b_tb_tile_base + stage * n + j_iter_tile_base;

          a_thread_tile_base_copy = a_tb_tile_offset;
          b_thread_tile_base_copy = b_tb_tile_offset;

          // We try to do the address calculation in the global memory with as
          // less address computation as possible. Hence, we compute the base
          // for each threads copy here and then just add some minimal offset
          // later.
          int4 *agmemBase = (int4 *)a_thread_tile_base_copy +
                            a_global_row * (K / 8) + a_global_col;
          int4 *asmemBase =
              (int4 *)&asmem[((stage / Ktile) % smem_stage_offset) *
                             (Mtile * (Ktile + PADDING_A))];
          asmemBase =
              asmemBase + a_smem_row * ((Ktile + PADDING_A) / 8) + a_smem_col;
          int4 *bgmemBase = (int4 *)b_thread_tile_base_copy +
                            b_global_row * (N / 8) + b_global_col;
          int4 *bsmemBase =
              (int4 *)&bsmem[((stage / Ktile) % smem_stage_offset) *
                             (Ktile * (Ntile + PADDING_B))];
          bsmemBase =
              bsmemBase + b_smem_row * ((Ntile + PADDING_B) / 8) + b_smem_col;

          // Acquire a new stage in the pipeline for every iteration of this
          // loop.
          pipe.producer_acquire();
#pragma unroll
          for (int i = 0, e = (Mtile * (Ktile / 8)) / numThreads; i < e; ++i) {
            cuda::memcpy_async(asmemBase, agmemBase, shape4, pipe);
            asmemBase = asmemBase + a_offset_per_thread_smem;
            agmemBase = agmemBase + a_offset_per_thread_gmem;
          }
#pragma unroll
          for (int i = 0, e = (Ktile * (Ntile / 8)) / numThreads; i < e; ++i) {
            cuda::memcpy_async(bsmemBase, bgmemBase, shape4, pipe);
            bsmemBase = bsmemBase + b_offset_per_thread_smem;
            bgmemBase = bgmemBase + b_offset_per_thread_gmem;
          }
          pipe.producer_commit();
        }
        // Wait for operations committed in all stages but the last `STAGES`
        // - 1. There will be `2 * (STAGES - 1)` items in the pipeline at a
        // time. We only need to wait for the `STAGES - 1` .
        cuda::pipeline_consumer_wait_prior<STAGES - 1>(pipe);
        __syncthreads();

        // warp_tile_base for compute is equal to the base address in shared
        // memory.
        a_warp_tile_base = &asmem[(((kk - ((STAGES - 1) * Ktile)) / Ktile) %
                                   smem_stage_offset) *
                                  (Mtile * (Ktile + PADDING_A))];
        b_warp_tile_base = &bsmem[(((kk - ((STAGES - 1) * Ktile)) / Ktile) %
                                   smem_stage_offset) *
                                  (Ktile * (Ntile + PADDING_B))];

#pragma unroll
        for (int stage = kk - ((STAGES == 1 ? 0 : (STAGES - 1)) * Ktile);
             stage < kk; stage += Ktile) {
          // warp_tile_base for compute is equal to the base address in shared
          // memory.
          a_warp_tile_base = &asmem[((stage / Ktile) % smem_stage_offset) *
                                    (Mtile * (Ktile + PADDING_A))];
          b_warp_tile_base = &bsmem[((stage / Ktile) % smem_stage_offset) *
                                    (Ktile * (Ntile + PADDING_B))];
          for (int kkk = 0; kkk < Ktile; kkk += WarpKtile) {
            if (kkk < k) {
              // Warp tile offset of asmem will only be dependent on the
              // warpIdx.y i.e., the row of the warp which is computing this
              // particular part. This points to the starting address of this
              // warp for the `a` operand.
              a_warp_tile_offset_compute =
                  a_warp_tile_base + (i_iter_warp_base * (Ktile + PADDING_A)) +
                  kkk;

              // Warp tile offset of bsmem will only be dependent on the
              // warpIdx.x i.e., the col of the warp which is computing this
              // particular part. This points to the starting address of this
              // warp for the `b` operand.
              b_warp_tile_offset_compute = b_warp_tile_base +
                                           (kkk * (Ntile + PADDING_B)) +
                                           (j_iter_warp_base);

              // This micro-kernel below re-uses the value of the A registers.
              // The compiler should have done this optimization but was not
              // able so we had to do this manually.
#pragma unroll
              for (int i = 0; i < WarpMtile; i += WM) {
                wmma::load_matrix_sync(
                    a_frag[i / WM],
                    a_warp_tile_offset_compute + (i * (Ktile + PADDING_A)),
                    (Ktile + PADDING_A));
              }
#pragma unroll
              for (int j = 0; j < WarpNtile; j += WN) {
                wmma::load_matrix_sync(b_frag, b_warp_tile_offset_compute + j,
                                       (Ntile + PADDING_B));
#pragma unroll
                for (int i = 0; i < WarpMtile; i += WM) {
                  wmma::mma_sync(c_accum[i / WM][j / WN], a_frag[i / WM],
                                 b_frag, c_accum[i / WM][j / WN]);
                }
              }
            }
          }
        }
      }

      // Wait for all data copy stages to complete.
      cuda::pipeline_consumer_wait_prior<0>(pipe);
      __syncthreads();

      // Remaining iterations for the k-loop.
#pragma unroll
      for (int kk = k - ((STAGES - 1) * Ktile); kk < k; kk += Ktile) {
        // warp_tile_base for compute is equal to the base address in shared
        // memory.
        a_warp_tile_base = &asmem[((kk / Ktile) % smem_stage_offset) *
                                  (Mtile * (Ktile + PADDING_A))];
        b_warp_tile_base = &bsmem[((kk / Ktile) % smem_stage_offset) *
                                  (Ktile * (Ntile + PADDING_B))];

#pragma unroll
        for (int kkk = 0; kkk < Ktile; kkk += WarpKtile) {
          if (kkk < k) {
            // Warp tile offset of asmem will only be dependent on the
            // warpIdx.y i.e., the row of the warp which is computing this
            // particular part. This points to the starting address of this
            // warp for the `a` operand.
            a_warp_tile_offset_compute =
                a_warp_tile_base + (i_iter_warp_base * (Ktile + PADDING_A)) +
                kkk;

            // Warp tile offset of bsmem will only be dependent on the
            // warpIdx.x i.e., the col of the warp which is computing this
            // particular part. This points to the starting address of this
            // warp for the `b` operand.
            b_warp_tile_offset_compute = b_warp_tile_base +
                                         (kkk * (Ntile + PADDING_B)) +
                                         (j_iter_warp_base);

            // This micro-kernel below re-uses the value of the A registers.
            // The compiler should have done this optimization but was not able
            // so we had to do this manually.
#pragma unroll
            for (int i = 0; i < WarpMtile; i += WM) {
              wmma::load_matrix_sync(
                  a_frag[i / WM],
                  a_warp_tile_offset_compute + (i * (Ktile + PADDING_A)),
                  (Ktile + PADDING_A));
            }
#pragma unroll
            for (int j = 0; j < WarpNtile; j += WN) {
              wmma::load_matrix_sync(b_frag, b_warp_tile_offset_compute + j,
                                     (Ntile + PADDING_B));
#pragma unroll
              for (int i = 0; i < WarpMtile; i += WM) {
                wmma::mma_sync(c_accum[i / WM][j / WN], a_frag[i / WM], b_frag,
                               c_accum[i / WM][j / WN]);
              }
            }
          }
        }
        pipe.consumer_release();
      }
    }
  }

// K-dimension processing of one warp is finished. We can copy the accumulator
// fragment corresponding to this warp to the result array `d` in global
// memory.
// TODO: currently assuming that one warp tile is only mapped to one warp,
// i.e., one warp needs to calculate only one warp tile. Hence
// d_warp_tile_offset is known form the last time it was calculated and hence
// reused here.
#pragma unroll
  for (int i = 0; i < WarpMtile; i += WM) {
#pragma unroll
    for (int j = 0; j < WarpNtile; j += WN) {
      wmma::store_matrix_sync(d_warp_tile_offset + ((i * N) + j),
                              c_accum[i / WM][j / WN], N, C_LAYOUT);
    }
  }
}

void hostGEMM(DTYPEAB *a, DTYPEAB *b, DTYPECD *c, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      DTYPECD temp = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        temp += __half2float(a[i * k + kk]) * __half2float(b[kk * n + j]);
      }
      c[i * n + j] = temp;
    }
  }
}

void compareGEMM(DTYPECD *h_c, DTYPECD *h_c_gpu_res, int m, int n) {
  int counter = 0;
  for (int i = 0; i < N * M; i++) {
    if (fabs(h_c_gpu_res[i] - h_c[i]) > 0.5f) {
      printf("DEBUG_GPU: mismatch i=%d result_Device=%f result_host=%f\n", i,
             h_c_gpu_res[i], h_c[i]);
      counter++;
    }
  }
  if (counter != 0)
    printf("DEBUG_CPU: Output does not match!: %d, %d\n", counter, m * n);
  else
    printf("DEBUG_CPU: Output matches!\n");
}

__global__ void compareGEMMOnDevice(DTYPECD *d_d, DTYPECD *d_d_naive, int m,
                                    int n) {
  int counter = 0;
  for (int i = 0; i < m * n; i++) {
    if (fabs(d_d[i] - d_d_naive[i]) > 0.5f) {
      ++counter;
    }
  }
  if (counter != 0)
    printf("DEBUG_GPU: Output does not match!: %d, %d\n", counter, m * n);
  else
    printf("DEBUG_GPU: Output matches!\n");
}

int main() {
  DTYPEAB *d_a, *d_b, *h_a, *h_b;
  DTYPECD *d_c, *d_d, *h_c, *h_c_gpu_res, *d_c_naive;
  int m, n, k;

  m = M;
  n = N;
  k = K;

  h_a = (DTYPEAB *)malloc(m * k * sizeof(DTYPEAB));
  h_b = (DTYPEAB *)malloc(k * n * sizeof(DTYPEAB));
  h_c = (DTYPECD *)malloc(m * n * sizeof(DTYPECD));
  h_c_gpu_res = (DTYPECD *)malloc(m * n * sizeof(DTYPECD));
  check_cuda_error(cudaMalloc(&d_a, m * k * sizeof(DTYPEAB)));
  check_cuda_error(cudaMalloc(&d_b, k * n * sizeof(DTYPEAB)));
  check_cuda_error(cudaMalloc(&d_c, m * n * sizeof(DTYPECD)));
  check_cuda_error(cudaMalloc(&d_d, m * n * sizeof(DTYPECD)));
  check_cuda_error(cudaMalloc(&d_c_naive, m * n * sizeof(DTYPECD)));

  assert(((unsigned long long)d_a) % 128 == 0);
  assert(((unsigned long long)d_b) % 128 == 0);
  assert(((unsigned long long)d_c) % 128 == 0);
  assert(((unsigned long long)d_d) % 128 == 0);

  init_host_matrices(h_a, h_b, h_c);
  check_cuda_error(
      cudaMemcpy(d_a, h_a, m * k * sizeof(DTYPEAB), cudaMemcpyHostToDevice));
  check_cuda_error(
      cudaMemcpy(d_b, h_b, k * n * sizeof(DTYPEAB), cudaMemcpyHostToDevice));
  check_cuda_error(
      cudaMemcpy(d_c, h_c, m * n * sizeof(DTYPECD), cudaMemcpyHostToDevice));

  dim3 block(NUM_THREADS_PER_BLOCK, 1, 1);
  dim3 grid((n + Ntile - 1) / Ntile, (m + Mtile - 1) / Mtile, 1);

  // Prefer shared memory config.
  unsigned long smem_capacity = std::max(
      (((Mtile * (Ktile + PADDING_A)) + (Ktile * (Ntile + PADDING_B))) *
       (STAGES == 1 ? 1 : (STAGES - 1) * 2)) *
          sizeof(DTYPEAB),
      Mtile * (Ntile + PADDING_C) * sizeof(DTYPECD));
  check_cuda_error(cudaFuncSetCacheConfig(GEMM, cudaFuncCachePreferShared));
  check_cuda_error(cudaFuncSetAttribute(
      GEMM, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_capacity));
  cout << "Using " << std::ceil(smem_capacity / 1024)
       << " KiB for shared memory\n";

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);
  int num_iters = 50;
  for (int i = 0; i < num_iters; ++i) {
    GEMM<<<grid, block, smem_capacity>>>(d_a, d_b, d_c, d_d, m, n, k);
  }
  cudaEventRecord(stop, NULL);

  cudaEventSynchronize(stop);
  float msecTotal = 0.0f;
  cudaEventElapsedTime(&msecTotal, start, stop);
  msecTotal = msecTotal / num_iters;
  double flopsPerMatrixMul = 2.0 * (double)m * (double)n * (double)k;
  double teraFlops = (flopsPerMatrixMul * 1.0e-12f) / (msecTotal / 1000.0f);
  cout << "Time: " << msecTotal << " ms\n";
  cout << "PERF: " << teraFlops << " Tflops\n";

  check_cuda_error(cudaPeekAtLastError());
  check_cuda_error(cudaDeviceSynchronize());
  cudaMemcpy(h_c_gpu_res, d_d, m * n * sizeof(DTYPECD), cudaMemcpyDeviceToHost);

#ifdef DEBUG_GPU
  dim3 block2(NBLOCK, MBLOCK, 1);
  dim3 grid2((n + NBLOCK - 1) / NBLOCK, (m + MBLOCK - 1) / MBLOCK, 1);

  GEMM_NAIVE<<<grid2, block2>>>(d_a, d_b, d_c_naive, m, n, k);

  check_cuda_error(cudaPeekAtLastError());
  check_cuda_error(cudaDeviceSynchronize());

  compareGEMMOnDevice<<<1, 1>>>(d_d, d_c_naive, m, n);
#endif

#ifdef DEBUG
  hostGEMM(h_a, h_b, h_c, m, n, k);

  compareGEMM(h_c, h_c_gpu_res, m, n);
#endif

#ifdef PRINT_HOST
  printMatrixFloat(h_c, m, n);
#endif

#ifdef PRINT_GPU
  printMatrixFloat(h_c_gpu_res, m, n);
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
