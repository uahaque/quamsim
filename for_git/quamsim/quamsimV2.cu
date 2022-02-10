#include <stdio.h>
#include <iostream>
#include <math.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector Multiplication of A and B into C. 
 */
__global__ void MatMul(const float *A, const float *B, float *C, int qubit, int qubit_val_2pow, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int bq = (i & (1 << qubit))>>qubit ;
    int i_plus_qubit;
    int i_minus_qubit;

    if(i < numElements)
    {
      if(bq == 0)
      {
        i_plus_qubit = i + qubit_val_2pow;
        C[i] = A[0]*B[i] + A[1]*B[i_plus_qubit];
      }  
      else if (bq == 1)
      {
        i_minus_qubit = i - qubit_val_2pow;
        C[i] = A[2]*B[i_minus_qubit] + A[3]*B[i];
      }  
    }
}

/**
 * Host main routine
 */
int main(int argc, char* argv[])
{
    FILE *FP, *FP_OUT;
    int count=0;
    char str[50];
    int qubit=0;
    int numElements=0;
    char *input_file;

    input_file = argv[1];

    FP = fopen(input_file, "r");

    if(FP==NULL) {
      // Throw error and exit if fopen() failed
      printf("Error: Unable to open file %s\n",input_file);
      exit(EXIT_FAILURE);
    }

    while (fscanf(FP,"%s", str) != EOF ){
      count++;
    }

    numElements = count - 5;

    FP = fopen(input_file, "r");

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    //int numElements;
    size_t size = numElements * sizeof(float);

    // Define input vectors A, B, C to be shared by host and device
    float *A, *B, *C;

    // Allocate unified Memory -- accessible from CPU and GPU
    cudaMallocManaged(&A,size);
    cudaMallocManaged(&B,size);
    cudaMallocManaged(&C,size);

    // Verify that allocations succeeded
    if (A == NULL || B == NULL || C == NULL)
    {
        fprintf(stderr, "Failed to allocate vectors!\n");
        exit(EXIT_FAILURE);
    }

    fscanf(FP,"%f %f", &A[0], &A[1]);
    fscanf(FP,"%f %f", &A[2], &A[3]);

    for(int i =0; i< numElements; i++){
      fscanf(FP,"%f",&B[i]);
    }

    fscanf(FP, "%d", &qubit);
    
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    int qubit_val_2pow = pow(2,qubit);
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    auto start = high_resolution_clock::now();
    MatMul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, qubit, qubit_val_2pow, numElements);
    auto end = high_resolution_clock::now();
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    FP_OUT = fopen("output.txt","w");

    if(FP_OUT == NULL) {
      // Throw error and exit if fopen() failed
      printf("Error: Unable to open file output.txt\n");
      exit(EXIT_FAILURE);
    }

    for(int i =0 ; i < numElements ; i++ ){
      //fprintf(FP_OUT,"%.3f\n", C[i]);
      printf("%.3f\n",C[i]);
    }

    auto duration = duration_cast<microseconds>(end - start);

    //printf("Time taken by kernel function : %d us\n",duration);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  
    // Free global memory
    err = cudaFree(A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

