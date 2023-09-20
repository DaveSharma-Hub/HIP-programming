#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

#define HIP_ASSERT(x){ \
	hipError_t status = x; \
	if(status!=hipSuccess){ \
		std::cerr <<"Error: "<<hipGetErrorString(status) << std::endl; \
		std::abort(); }}

using namespace std;


__global__ void matrixMultRowWise(int** first, int** second,int** result, int colCount, int rowCount){
    int id = threadIdx.x * blockIdx.x * blockDim.x;
    if(id<rowCount){
        for(int i=0;i<colCount;i++){
            result[id][i] = first[id][i] + second[id][i];
        }
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

    int blockSize = 1024;
    int gridSize = (int)ceil((float)rowCount/blockSize);

    cpuA = (int**)malloc((sizeof(int*)*rowCount));
    cpuB = (int**)malloc((sizeof(int*)*rowCount));
    cpuC = (int**)malloc((sizeof(int*)*rowCount));
    cpuCVerify = (int**)malloc((sizeof(int*)*rowCount));
    
    HIP_ASSERT(hipMalloc(&gpuA,(sizeof(int*)*rowCount)));
    HIP_ASSERT(hipMalloc(&gpuB,(sizeof(int*)*rowCount)));
    HIP_ASSERT(hipMalloc(&gpuC,(sizeof(int*)*rowCount)));


    for(int i=0;i<rowCount;i++){
        cpuA[i] = (int*)malloc(colSize);
        cpuB[i] = (int*)malloc(colSize);
        cpuC[i] = (int*)malloc(colSize);
        cpuCVerify[i] = (int*)malloc(colSize);

        // HIP_ASSERT(hipMalloc(&gpuA[i],colSize));
        // HIP_ASSERT(hipMalloc(&gpuB[i],colSize));
        // HIP_ASSERT(hipMalloc(&gpuC[i],colSize));
    }

    for(int i=0;i<rowCount;i++){
        for(int j=0;j<colCount;j++){
            cpuA[i][j] = 1;
            cpuB[i][j] = 2;
            cpuCVerify[i][j] = cpuA[i][j] + cpuB[i][j];
        }
    }

    // HIP_ASSERT(hipMemcpy(gpuA,cpuA,rowSize,hipMemcpyHostToDevice));
    // HIP_ASSERT(hipMemcpy(gpuB,cpuB,rowSize,hipMemcpyHostToDevice));
    // for(int i=0;i<colCount;i++){
    //     HIP_ASSERT(hipMemcpy(gpuA[i],cpuA[i],colSize,hipMemcpyHostToDevice));
    //     HIP_ASSERT(hipMemcpy(gpuB[i],cpuB[i],colSize,hipMemcpyHostToDevice));
    // }


    // matrixMultRowWise<<<gridSize, blockSize>>>(gpuA, gpuB, gpuC,colCount, rowCount);
    // HIP_ASSERT(hipDeviceSynchronize());

    // HIP_ASSERT(hipMemcpy(cpuC,gpuC,rowSize,hipMemcpyDeviceToHost));
    // for(int i=0;i<colCount;i++){
    //     HIP_ASSERT(hipMemcpy(cpuC[i],gpuC[i],colSize,hipMemcpyDeviceToHost));
    // }

    // // validate
    // for(int i=0;i<rowCount;i++){
    //     for(int j=0;j<colCount;j++){
    //         int gpuValue = cpuC[i][j];
    //         int cpuValue = cpuCVerify[i][j];
    //         if(abs(gpuValue-cpuValue)>1e-5){
    //             printf("Error value at index row=%d col=%d, expected %d recieved %d\n",i+1,j+1,cpuValue, gpuValue);
    //         }
    //     }
    // }


    for(int i=0;i<rowCount;i++){
        free(cpuA[i]);
        free(cpuB[i]);
        free(cpuC[i]);
        free(cpuCVerify[i]);

        // HIP_ASSERT(hipFree(gpuA[i]));
        // HIP_ASSERT(hipFree(gpuB[i]));
        // HIP_ASSERT(hipFree(gpuC[i]));
    }

    HIP_ASSERT(hipFree(gpuA));
    HIP_ASSERT(hipFree(gpuB));
    HIP_ASSERT(hipFree(gpuC));
    free(cpuA);
    free(cpuB);
    free(cpuC);
    free(cpuCVerify);
}