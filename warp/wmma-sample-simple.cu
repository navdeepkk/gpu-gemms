#include "common.h"
#include <stdio.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
using namespace std;
__global__ void test_wmma(__half * da){
  //wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag;
  
  //wmma::load_matrix_sync( frag, da, 16 );

  asm(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      " { %0, %1 }, "
      " { %2 }, "
      " { %3 }, "
      " { %4, %5 }; "
      :
      "=r"(d[cd_idx]), "=r"(d[cd_idx + 1])
      :
      "r"(a[ab_idx]),
      "r"(b[ab_idx]),
      "r"(c[cd_idx]), "r"(c[cd_idx + 1])
     );
  
  for(int i = 0 ; i < 16; ++i){
    float f = __half2float(frag[0].x[i]);
    printf("%f ", f);
  }
  
  //wmma::store_matrix_sync( da, a_frag, 16, wmma::mem_row_major);
}

void init_matrix(__half * ha, int m, float init){
  for(int i = 0; i < m; ++i){
    for(int j = 0; j < m; ++j){
      ha[i * 16 + j] = __float2half(init);
    }
  }
}

void printFrag(__half * ha, int m){
  for(int i = 0; i < m; ++i){
    printf("%f", __half2float(ha[i]));
  }
  printf("\n");
}

void TestWMMA()
{
  int threads = 32;
  int blocks = 1;

  __half *da, *ha;
  ha = (__half*)malloc(16 * 16 * sizeof(__half));
  // Initialize with ones.
  init_matrix(ha, 16, 1.0f);
  cudaMalloc(&da, 16 * 16 * sizeof(__half));
  cudaMemcpy(da, ha, 16 * 16 * sizeof(__half), cudaMemcpyHostToDevice);

  // replace with 0's.
  init_matrix(ha, 16, 0.0f);

  test_wmma<<<blocks, threads>>>(da);
  check_cuda_error(cudaPeekAtLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(ha, da, 16 * 16 * sizeof(__half), cudaMemcpyDeviceToHost);

  //for(int i = 0; i < 16; ++i){
  //  printFrag(a_frag[i].x, 16);
  //}
}

int main(){
  TestWMMA();
  return 0;
}
