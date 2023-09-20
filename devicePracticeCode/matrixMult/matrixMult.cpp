#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

#define HIP_ASSERT(x) (assert((x)==hipSuccess));

using namespace std;

__global__ matrixMult(int** first, int** second, int** result, int firstRow, int firstCol){
    int x = threadIdx.x * blockDim.x * blockIdx.x;
    int y = threadIdx.y * blockDim.y * blockIdx.y;

    int size = firstCol; // equivalent to secondRow;
    if(y<firstRow && x<firstCol){
        int sum = 0;
        for(int i=0;i<size;i++){
            sum += first[y][i] * second[i][x];
        }
        result[x][y] = sum;
    }   
}

int main(){
    
    int m;
    int n;

    int** a; // m x n matrix
    int** b; // n x m matrix
    int** c; //  c = a * b
    int** verifyC;

    size_t mSize = sizeof(int) * m;
    size_t nSize = sizeof(int) * n;

    malloc(a,mSize);
    malloc(b,nSize);
    malloc(c,mSize);
    malloc(verifyC,mSize);

    int** gpuA;
    int** gpuB;
    int** gpuC;
    
    HIP_ASSERT(hipMalloc(&gpuA,mSize));
    HIP_ASSERT(hipMalloc(&gpuB,nSize));
    HIP_ASSERT(hipMalloc(&gpuC,mSize));


    for(int i=0;i<mSize;i++){
        malloc(a[i],nSize);
        malloc(c[i],mSize);
        malloc(verifyC[i],mSize);

        HIP_ASSERT(hipMalloc(&gpuA[i],nSize));
        HIP_ASSERT(hipMalloc(&gpuC[i],mSize));
    }

    for(int i=0;i<nSize;i++){
        malloc(b[i],mSize);
        HIP_ASSERT(hipMalloc(&gpuB[i],mSize));
    }

    for(int i=0;i<mSize;i++){
        for(int j=0;j<nSize;j++){
            a[i][j] = 3;
            b[j][i] = 2
        }
    }

    // may not need below
    HIP_ASSERT(hipMemcpy(gpuA, a, mSize, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(gpuB, b, nSize, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(gpuC, c, mSize, hipMemcpyHostToDevice));

    for(int i=0;i<mSize;i++){
        HIP_ASSERT(hipMemcpy(gpuA[i], a[i], nSize, hipMemcpyHostToDevice));
        HIP_ASSERT(hipMemcpy(gpuC[i], c[i], mSize, hipMemcpyHostToDevice));
    }

    for(int j=0;j<nSize;j++){
        HIP_ASSERT(hipMemcpy(gpuB[i], b, mSize, hipMemcpyHostToDevice));
    }

    matrixMult<<<gridSize,blockSize>>>(gpuA, gpuB,gpuC,m,n);

    HIP_ASSERT(hipDeviceSynchronize());
    for(int i=0;i<mSize;i++){
        HIP_ASSERT(hipMemcpy(c[i], gpuC[i],mSize,hipMemcpyDeviceToHost));
    }

    //compare;

    for(int i=0;i<mSize;i++){
        HIP_ASSERT(hipFree(gpuA[i]));
        HIP_ASSERT(hipFree(gpuC[i]));

        free(a[i]);
        free(c[i]);
        free(verifyC[i]);
    }

    for(int j=0;j<nSize;j++){
        HIP_ASSERT(hipFree(gpuB[j]));
        free(b[j]);
    }

    HIP_ASSERT(hipFree(gpuA));
    HIP_ASSERT(hipFree(gpuB));
    HIP_ASSERT(hipFree(gpuC));

    free(a);
    free(b);
    free(c);
    free(verifyC);

}