#include "common.h"
#include<stdio.h>
#include<cuda.h>
#define height 50
#define width 50

// Device code
__global__ void kernel(float** devPtr, int pitch)
{
  float a = devPtr[1][1];
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      float element = a;
    }
  }
}

//Host Code
int main()
{

  cudaMalloc((void**)&ppArray_a, 10 * sizeof(int*));

  for(int i=0; i<10; i++){
    cudaMalloc(&someHostArray[i], 100*sizeof(int)); /* Replace 100 with the dimension that u want */
  }

  cudaMemcpy(ppArray_a, someHostArray, 10*sizeof(int *), cudaMemcpyHostToDevice);

  float** devPtr;
  int r = 3, c = 3;

  check_cuda_error(cudaMalloc(&devPtr, r * sizeof(float *)));

  for (int i = 0; i < r; i++)
    check_cuda_error(cudaMalloc( devPtr + i, c * sizeof(float)));

  //size_t pitch;
  //cudaMallocPitch((void**)&devPtr, &pitch, width * sizeof(float), height);
  //kernel<<<100, 512>>>(devPtr, pitch);
  return 0;
}
