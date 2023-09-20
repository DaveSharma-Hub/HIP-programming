#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>

#define HIP_ASSERT(x){ \
	hipError_t status = x; \
	if(status!=hipSuccess){ \
		std::cerr <<"Error: "<<hipGetErrorString(status) << std::endl; \
		std::abort(); }}

using namespace std;


__global__ void matrixMultRowWise(int* first, int* second,int* result, int colCount, int rowCount){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id<rowCount){
        for(int i=0;i<colCount;i++){
            result[(colCount* id) + i] = first[(colCount * id) + i] + second[(colCount * id) + i];
        }
    }
}

void test(int rowCount, int colCount, int blockSize){
    int* cpuA;
    int* cpuB;
    int* cpuC;
    int* cpuCVerify;

    int* gpuA;
    int* gpuB;
    int* gpuC;

    // int rowCount = 100;
    // int colCount = 50;

    size_t flattenedSize = rowCount * colCount * sizeof(int);

    // int blockSize = 1024;
    int gridSize = (int)ceil((float)rowCount/blockSize);

    cpuA = (int*)malloc(flattenedSize);
    cpuB = (int*)malloc(flattenedSize);
    cpuC = (int*)malloc(flattenedSize);
    cpuCVerify = (int*)malloc(flattenedSize);
    
    HIP_ASSERT(hipMalloc(&gpuA,(flattenedSize)));
    HIP_ASSERT(hipMalloc(&gpuB,(flattenedSize)));
    HIP_ASSERT(hipMalloc(&gpuC,(flattenedSize)));


    for(int i=0;i<rowCount;i++){
        for(int j=0;j<colCount;j++){
            cpuA[(colCount * i) + j] = 1;
            cpuB[(colCount * i) + j ] = 2;
        }
    }

    auto t_startCPU = std::chrono::high_resolution_clock::now();

    for(int i=0;i<rowCount;i++){
        for(int j=0;j<colCount;j++){
            cpuCVerify[(colCount * i) + j] = cpuA[(colCount * i) + j] + cpuB[(colCount * i) + j];
        }
    }

    auto t_endCPU = std::chrono::high_resolution_clock::now();

    double elapsed_time_ns_CPU = std::chrono::duration<double, std::nano>(t_endCPU-t_startCPU).count();


    HIP_ASSERT(hipMemcpy(gpuA,cpuA,flattenedSize,hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(gpuB,cpuB,flattenedSize,hipMemcpyHostToDevice));

    auto t_startGPU = std::chrono::high_resolution_clock::now();
    matrixMultRowWise<<<gridSize, blockSize>>>(gpuA, gpuB, gpuC, colCount, rowCount);
    auto t_endGPU = std::chrono::high_resolution_clock::now();
    HIP_ASSERT(hipDeviceSynchronize());

    double elapsed_time_ns_GPU = std::chrono::duration<double, std::nano>(t_endGPU-t_startGPU).count();


    HIP_ASSERT(hipMemcpy(cpuC,gpuC,flattenedSize,hipMemcpyDeviceToHost));

    // validate
    for(int i=0;i<rowCount;i++){
        for(int j=0;j<colCount;j++){
            int gpuValue = cpuC[(colCount * i) + j];
            int cpuValue = cpuCVerify[(colCount * i) + j];
            if(abs(gpuValue-cpuValue)>1e-5){
                printf("Error value at index row=%d col=%d, expected %d recieved %d\n",i+1,j+1,cpuValue, gpuValue);
            }
        }
    }

    cout<<"GPU time: "<<elapsed_time_ns_GPU<<" ns\n";
    cout<<"CPU time: "<<elapsed_time_ns_CPU<<" ns\n";

    HIP_ASSERT(hipFree(gpuA));
    HIP_ASSERT(hipFree(gpuB));
    HIP_ASSERT(hipFree(gpuC));
    free(cpuA);
    free(cpuB);
    free(cpuC);
    free(cpuCVerify);
}


int main(){
    cout<<"Order magnitude test"<<endl;
    for(int i=1000;i<100000;i*=10){
        int rowCount = i;
        int colCount = i/2;
        cout<<"\n---------------------\nRow size: "<<rowCount<<" Col size: "<<colCount<<endl;
        test(rowCount, colCount, 1024);
    }

    cout<<"\nBlock Size test multiple of 64: "<<endl;
    int rowCount = 10000;
    int colCount = 5000;
    for(int i=64;i<=1024;i+=64){
        int blockSize = i;
        cout<<"\n---------------------\nBlock size: "<<blockSize<<endl;
        test(rowCount, colCount, blockSize);
    }
    cout<<"\nBlock Size test not multiple of 64: "<<endl;
    for(int i=50;i<=1024;i+=50){
        int blockSize = i;
        cout<<"\n---------------------\nBlock size: "<<blockSize<<endl;
        test(rowCount, colCount, blockSize);
    }
}