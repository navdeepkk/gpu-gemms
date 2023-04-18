// This file includes the following GEMM parameters:-
// 1.) Problem size,
// 2.) Thread block tile sizes,
// 3.) Register/warp tile sizes,
// 4.) Shared memory padding for the shared memory buffers,
// 5.) Stages of pipelining.

#define M 1024
#define N 3072
#define K 768
#define WM 16
#define WN 16
#define WK 16
#define Mtile 128  // This will actually be the loop step of `i` loop.
#define Ntile 128  // This will actually be the loop step of `j` loop.
#define Ktile 32   // This will actually be the loop step of `k` loop.
#define WarpMtile 64
#define WarpNtile 64
#define WarpKtile \
  16  // 16 because the size supported by the wmma api is 16x16x16.
#define PADDING_A 8
#define PADDING_B 8
#define PADDING_C 4
#define STAGES 2
