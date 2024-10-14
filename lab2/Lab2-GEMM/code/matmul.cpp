// g++ matmul.cpp -o matmul -std=c++17 -O3 -Wall && ./matmul

#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>


double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return 1e6 * tv.tv_sec + tv.tv_usec;
}

constexpr int n = 512;
int A[n][n];
int B[n][n];
int BT[n][n];
int AT[n][n];
int C[n][n];
int C_groundtruth[n][n];

void init() {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = rand(); 
      B[i][j] = rand(); 
    } 
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C_groundtruth[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void test() {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      assert(C[i][j] == C_groundtruth[i][j]);
    }
  }
}

void matmul() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}

void matmulUnrolled() {
  memset(C, 0, sizeof(C));
  int i, j, k;
  int unroll_factor = 4; // Unrolling by a factor of 4

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k <= n - unroll_factor; k += unroll_factor) {
        C[i][j] += A[i][k] * B[k][j];
        C[i][j] += A[i][k + 1] * B[k + 1][j];
        C[i][j] += A[i][k + 2] * B[k + 2][j];
        C[i][j] += A[i][k + 3] * B[k + 3][j];
      }
      // Handle the remaining elements
      for (; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

void matmulWriteCaching() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int temp = 0; // Cache for C[i][j]
      for (int k = 0; k < n; k++) {
        temp += A[i][k] * B[k][j];    
      }
      C[i][j] = temp; // Write back the cached value
    }
  }
}

void matmulTiled() {
  memset(C, 0, sizeof(C));
  int tile_size = 16; // Define the tile size

  for (int ii = 0; ii < n; ii += tile_size) {
    for (int jj = 0; jj < n; jj += tile_size) {
      for (int kk = 0; kk < n; kk += tile_size) {
        // Multiply the tiles
        for (int i = ii; i < ii + tile_size && i < n; i++) {
          for (int j = jj; j < jj + tile_size && j < n; j++) {
            int temp = 0; // Cache for C[i][j]
            for (int k = kk; k < kk + tile_size && k < n; k++) {
              temp += A[i][k] * B[k][j];
            }
            C[i][j] += temp; // Write back the cached value
          }
        }
      }
    }
  }
}

/*
#include <immintrin.h> // Header for SIMD intrinsics

void matmulVector() {
  memset(C, 0, sizeof(C));
  int vector_size = 4; // Assuming we are using AVX with 256-bit registers (4 floats)

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      __m128 c_vec = _mm_setzero_ps(); // Initialize vector for C[i][j]
      int k;
      for (k = 0; k <= n - vector_size; k += vector_size) {
        __m128 a_vec = _mm_loadu_ps(&A[i][k]); // Load 4 elements from A[i][k]
        __m128 b_vec = _mm_loadu_ps(&B[k][j]); // Load 4 elements from B[k][j]
        c_vec = _mm_add_ps(c_vec, _mm_mul_ps(a_vec, b_vec)); // Multiply and accumulate
      }
      // Handle the remaining elements
      float temp[4];
      _mm_storeu_ps(temp, c_vec);
      C[i][j] += temp[0] + temp[1] + temp[2] + temp[3];
      for (; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}*/

void matmulPacked() {
  memset(C, 0, sizeof(C));
  float packedA[n][n];
  float packedB[n][n];

  // Pack matrix A
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
      packedA[i][k] = A[i][k];
    }
  }

  // Pack matrix B
  for (int k = 0; k < n; k++) {
    for (int j = 0; j < n; j++) {
      packedB[k][j] = B[k][j];
    }
  }

  // Perform matrix multiplication using packed arrays
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += packedA[i][k] * packedB[k][j];
      }
    }
  }
}

void matmul_ikj() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < n; j++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}

void matmul_AT() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      AT[i][j] = A[j][i];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += AT[k][i] * B[k][j];    
      }   
    }
  }
}

void matmul_BT() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      BT[i][j] = B[j][i];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * BT[j][k];    
      }   
    }
  }
}

/*
void matmul_BT() {
  constexpr int blockSize = 128; // 假設 blockSize 是 64，可以根據實際情況調整
  memset(C, 0, sizeof(C));
  
  // 轉置矩陣 B
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      BT[j][i] = B[i][j];
    }
  }

  // 使用分塊技術進行矩陣乘法
  for (int ii = 0; ii < n; ii += blockSize) {
    for (int jj = 0; jj < n; jj += blockSize) {
      for (int kk = 0; kk < n; kk += blockSize) {
        for (int i = ii; i < ii + blockSize && i < n; i++) {
          for (int j = jj; j < jj + blockSize && j < n; j++) {
            int sum = 0;
            for (int k = kk; k < kk + blockSize && k < n; k++) {
              sum += A[i][k] * BT[j][k];
            }
            C[i][j] += sum;
          }
        }
      }
    }
  }
}

void matmul_BT() {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      BT[i][j] = B[j][i];
    }
  }
  memset(C, 0, sizeof(C));
  

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
      for (int k = 0; k < n; k += 8) {
        sum0 += A[i][k] * BT[j][k];
        sum1 += A[i][k + 1] * BT[j][k + 1];
        sum2 += A[i][k + 2] * BT[j][k + 2];
        sum3 += A[i][k + 3] * BT[j][k + 3];
        sum4 += A[i][k + 4] * BT[j][k + 4];
        sum5 += A[i][k + 5] * BT[j][k + 5];
        sum6 += A[i][k + 6] * BT[j][k + 6];
        sum7 += A[i][k + 7] * BT[j][k + 7];
      }
      C[i][j] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
    }
  }
}

void matmul_BT() {
  constexpr int blockSize = 64; // 假設 blockSize 是 64，可以根據實際情況調整
  memset(C, 0, sizeof(C));
  
  // 轉置矩陣 B
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      BT[j][i] = B[i][j];
    }
  }

  // 使用分塊技術進行矩陣乘法
  for (int ii = 0; ii < n; ii += blockSize) {
    for (int jj = 0; jj < n; jj += blockSize) {
      for (int kk = 0; kk < n; kk += blockSize) {
        for (int i = ii; i < ii + blockSize && i < n; i++) {
          for (int j = jj; j < jj + blockSize && j < n; j++) {
            int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            for (int k = kk; k < kk + blockSize && k < n; k += 4) {
              sum0 += A[i][k] * BT[j][k];
              sum1 += A[i][k + 1] * BT[j][k + 1];
              sum2 += A[i][k + 2] * BT[j][k + 2];
              sum3 += A[i][k + 3] * BT[j][k + 3];
            }
            C[i][j] += sum0 + sum1 + sum2 + sum3;
          }
        }
      }
    }
  }
}
*/

int main() {
  init();
  float avg_time = 0.0f;
  for (int K = 0; K < 32; K++) {
    auto t = get_time();
    matmul();
    matmulUnrolled();
    matmulWriteCaching();
    matmulTiled();
    //matmulVector();
    matmulPacked();
    //matmul_ikj();
    //matmul_AT();
    //matmul_BT();
    printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
    test();
  }
  printf("Avg Time for Calculation: %f us\n", avg_time / 32);
  return 0;
}

