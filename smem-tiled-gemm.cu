#include<iostream>
#include<cuda_runtime.h>
#include<cuda.h>
#include<device_launch_parameters.h>
#define DTYPE float
#define M 1024 	 
#define N 1024
#define K 1024
#define Mtile 32
#define Ntile 32
#define Ktile 32

using namespace std;

__global__ void GEMM(DTYPE * a, DTYPE * b, DTYPE * c, int m, int n, int k){
  __shared__ DTYPE asmem[Mtile][Ktile];
  __shared__ DTYPE bsmem[Ktile][Ntile];

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  // Copy from GMEM to SMEM.
  // If ktile != ntile then all the threads must not participate in data copying
  // Ussually ktile is less than mtile or ntile. Currenlty handeling the case only when ktile is
  // less than mtile and ntile. In this case all the threads in the tile need not participate in the
  // data copying.

  int min_k_n = min(Ntile, Ktile);
  int min_k_m = min(Mtile, Ktile);
  DTYPE sum = 0;

  for(int kk = 0; kk < (k + Ktile - 1)/Ktile; ++kk){
    if(threadIdx.x < (unsigned int) min_k_n){
      asmem[threadIdx.y][threadIdx.x] = a[(row * k) + (kk * Ktile) + threadIdx.x];			
    }
    if(threadIdx.y < (unsigned int) min_k_m){
      bsmem[threadIdx.y][threadIdx.x] = b[(kk * Ktile + threadIdx.y) * n + col];			
    }

    __syncthreads();
    
    // Data into SMEM is loaded lets do the matmul for this tile.
    for(int kkk = 0; kkk < Ktile; ++kkk){
      if(threadIdx.x < (unsigned int) min_k_n && threadIdx.y < (unsigned int) min_k_m){
	sum += asmem[threadIdx.y][kkk] * bsmem[kkk][threadIdx.x];
      }	
    }
    __syncthreads();				
  }
  // directly storing to the global memory.
  c[row * n + col] = sum;
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
      if(abs(h_c[i * n + j] - h_c_gpu_res[i * n + j]) > 1e-9)
	return false;	
    }
  }
  return true;
}

void initMatrix(DTYPE * matrix, int m, int n){
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      matrix[i * n + j] = rand() % 100;
    }
  }
}

void printMatrix(DTYPE * matrix, int m, int n){
  for(int i = 0; i < n; ++i){
    for(int j = 0; j < m; ++j){
      cout<<matrix[i * n + j]<<" ";
    }
    cout<<endl;
  }
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

  dim3 block(Ntile, Mtile ,1);
  dim3 grid((n + Ntile - 1) / Ntile, (m + Mtile - 1) / Mtile , 1);

  GEMM<<<grid, block>>>(d_a, d_b, d_c, m , n, k);	
  cudaDeviceSynchronize();
  cudaMemcpy(h_c_gpu_res, d_c, m * n * sizeof(DTYPE), cudaMemcpyDeviceToHost);
  hostGEMM(h_a, h_b, h_c, m, n, k);

  //cout<<"cpu res: \n";
  //printMatrix(h_c, m ,n);
  //cout<<"gpu res: \n";
  //printMatrix(h_c_gpu_res, m ,n);
  
  cout<<compareGEMM(h_c, h_c_gpu_res, m, n)<<endl;

  free(h_a);
  free(h_b);
  free(h_c);
  free(h_c_gpu_res);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
