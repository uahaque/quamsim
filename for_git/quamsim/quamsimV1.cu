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
    size_t size = numElements * sizeof(float);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    fscanf(FP,"%f %f", &h_A[0], &h_A[1]);
    fscanf(FP,"%f %f", &h_A[2], &h_A[3]);

    for(int i =0; i< numElements; i++){
      fscanf(FP,"%f",&h_B[i]);
    }

    fscanf(FP, "%d", &qubit);
    
    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    int qubit_val_2pow = pow(2,qubit);
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    auto start = high_resolution_clock::now();
    MatMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, qubit, qubit_val_2pow, numElements);
    auto end = high_resolution_clock::now();
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    FP_OUT = fopen("output.txt","w");

    if(FP_OUT == NULL) {
      // Throw error and exit if fopen() failed
      printf("Error: Unable to open file output.txt\n");
      exit(EXIT_FAILURE);
    }

    for(int i =0 ; i < numElements ; i++ ){
      //fprintf(FP_OUT,"%.3f\n", h_C[i]);
      printf("%.3f\n",h_C[i]);
    }

    auto duration = duration_cast<microseconds>(end - start);

    //printf("Time taken by kernel function : %d us\n",duration);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  
    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

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

