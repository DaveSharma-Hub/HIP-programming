#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

#define HIP_ASSERT(x) (assert((x)==hipSuccess));

using namespace std;

__global__ matrixAddElementWise(int** first, int** second, int** result, int rowCount, int colCount){
    int xId = blockIdx.x * blockDim.x * threadIdx.x; 
    int yId = blockIdx.y * blockDim.y * threadIdx.y; 

    if(xId<rowCount && yId<colCount){
        result[xId][yId] = first[xId][yId] + second[xId][yId];
    }
}   

int main(){
    int** cpuA;
    int** cpuB;
    int** cpuC;
    int** cpuCVerify;

    int** gpuA;
    int** gpuB;
    int** gpuC;

    int rowCount = 10000;
    int colCount = 5000;

    size_t rowSize = rowCount * sizeof(int);
    size_t colSize = colCount * sizeof(int);

    dim3 blockSize = {1024,1024,1};
    dim3 gridSize = {(int)ceil((float)rowCount/blockSize.x),(int)ceil((float)colCount/blockSize.y)};

    malloc(cpuA,rowSize);
    malloc(cpuB,rowSize);
    malloc(cpuC,rowSize);
    malloc(cpuCVerify,rowSize);
    
    HIP_ASSERT(hipMalloc(&gpuA,rowSize));
    HIP_ASSERT(hipMalloc(&gpuB,rowSize));
    HIP_ASSERT(hipMalloc(&gpuC,rowSize));

    for(int i=0;i<colCount;i++){
        malloc(cpuA[i],colSize);
        malloc(cpuB[i],colSize);
        malloc(cpuC[i],colSize);
        malloc(cpuCVerify[i],colSize);

        HIP_ASSERT(hipMalloc(&gpuA[i],colSize));
        HIP_ASSERT(hipMalloc(&gpuB[i],colSize));
        HIP_ASSERT(hipMalloc(&gpuC[i],colSize));
    }

    for(int i=0;i<rowCount,i++){
        for(int j=0;j<colCount;j++){
            cpuA[i][j] = 1;
            cpuB[i][j] = 2;
            cpuCVerify[i][j] = cpuA[i][j] + cpuB[i][j];
        }
    }

    HIP_ASSERT(hipMemcpy(gpuA,cpuA,rowSize,hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(gpuB,cpuB,rowSize,hipMemcpyHostToDevice));
    for(int i=0;i<colCount;i++){
        HIP_ASSERT(hipMemcpy(gpuA[i],cpuA[i],colSize,hipMemcpyHostToDevice));
        HIP_ASSERT(hipMemcpy(gpuB[i],cpuB[i],colSize,hipMemcpyHostToDevice));
    }


    matrixAddElementWise<<<gridSize, blockSize>>>(gpuA, gpuB, gpuC,rowCount, colCount);
    HIP_ASSERT(hipDeviceSynchronize());

    HIP_ASSERT(hipMemcpy(cpuC,gpuC,rowSize,hipMemcpyDeviceToHost));
    for(int i=0;i<colCount;i++){
        HIP_ASSERT(hipMemcpy(cpuC[i],gpuC[i],colSize,hipMemcpyDeviceToHost));
    }

    // validate
    for(int i=0;i<rowCount;i++){
        for(int j=0;j<colCount;j++){
            int gpuValue = cpuC[i][j];
            int cpuValue = cpuCVerify[i][j];
            if(abs(gpuValue-cpuValue)>1e-5){
                printf("Error value at index row=%d col=%d, expected %d recieved %d\n",i+1,j+1,cpuValue, gpuValue);
            }
        }
    }


    for(int i=0;i<colCount;i++){
        free(cpuA[i]);
        free(cpuB[i]);
        free(cpuC[i]);
        free(cpuCVerify[i]);

        HIP_ASSERT(hipFree(gpuA[i]));
        HIP_ASSERT(hipFree(gpuB[i]));
        HIP_ASSERT(hipFree(gpuC[i]));
    }

    HIP_ASSERT(hipFree(gpuA));
    HIP_ASSERT(hipFree(gpuB));
    HIP_ASSERT(hipFree(gpuC));
    free(cpuA);
    free(cpuB);
    free(cpuC);
    free(cpuCVerify);
}   