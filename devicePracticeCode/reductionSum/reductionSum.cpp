#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

using namespace std;

const static int BLOCKSIZE = 256;

__global__ void reductionSum(const float* input, float* output, int size){

    int gridSize = blockDim.x * gridDim.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float sum = 0;
    for(int i=idx;i<size;i+=gridSize){
        sum += input[i];
    }

    __shared__ float localSum[BLOCKSIZE];
    localSum[threadIdx.x] = sum;
    __syncthreads();

    for(int s=BLOCKSIZE/2;s>0;s/=2){
        if(threadIdx.x<s){
            localSum[threadIdx.x] += localSum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        output[blockIdx.x] = localSum[0];
    }
}

float helperFunction(vector<float>& array,float sum, int beg, int end){
    if(beg==end){
        return array[beg] + sum;
    }else{
        int mid = (beg+end)/2;
        float sum1 = helperFunction(array, sum, beg, mid);
        float sum2 = helperFunction(array, sum, mid+1, end);
        return sum1 + sum2;
    }
}

float reduceSumArrayCPU(vector<float>& array){
    return helperFunction(array, 0, 0, array.size());
}

int main(){
    const static int N = 10485760;
    const static int num_blocks = 1200;

    vector<float> a(N);
    vector<float> b(num_blocks);

    for(int i=0;i<N;i++){
        a[i] = (float)rand();
    }

    float* gpuA;
    float* gpuB;

    HIP_ASSERT(hipMalloc(&gpuA, N*sizeof(float)));
    HIP_ASSERT(hipMalloc(&gpuB, num_blocks*sizeof(float)));

    HIP_ASSERT(hipMemcpy(gpuA, a.data(), a.size()*sizeof(float), hipMemcpyHostToDevice));
    
    reductionSum<<<num_blocks, BLOCKSIZE>>>(gpuA, gpuB, N);
    HIP_ASSERT(hipDeviceSynchronize());
    HIP_ASSERT(hipMemcpy(b.data(),gpuB, num_blocks*sizeof(float), hipMemcpyDeviceToHost));

    float sum = 0;
    for(int i=0;i<num_blocks;i++){
        sum += b[i];
    }

    // float expectedSum = 0;
    // for(int i=0;i<N;i++){
    //     expectedSum += a[i];
    // }

    float expectedSum = reduceSumArrayCPU(a);

    if(abs(sum-expectedSum)>1e-5){
        printf("Error: incorrect sum, expected: %lf, recieved: %lf\n",expectedSum, sum);
    }
}