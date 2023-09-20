#include <hip/hip_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

using namespace std;
using namespace cv;

__global__ void fixedGammaCorrection(uchar* imageData, float gamma, int num_values){
    int globalSize = blockDim.x * gridDim.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    for(;id<num_values;id+=globalSize){
        float normalizedPixelValue = imageData[id] / 255.0f;
        normalizedPixelValue = pow(normalizedPixelValue,gamma);
        imageData[id] = (uchar)(normalizedPixelValue * 255);
    }
}


void test(int gridSize, int blockSize){
    string inputFile = "./image2.jpg";
    string outputFile = "output"+ to_string(gridSize) + ".jpg" ;
    Mat image = imread(inputFile, IMREAD_COLOR);
    
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    float gamma  = 3.4;

    int imageDataSize = width*height*channels*sizeof(uchar);

    uchar* imageData = (uchar*)malloc(imageDataSize);
    uchar* gpuData;
    uchar* cpuData = (uchar*)malloc(imageDataSize);

    memcpy(imageData, image.data, imageDataSize);

    HIP_ASSERT(hipMalloc(&gpuData,imageDataSize));
    HIP_ASSERT(hipMemcpy(gpuData, imageData, imageDataSize, hipMemcpyHostToDevice));
    
    auto t_startGPU = std::chrono::high_resolution_clock::now();
    fixedGammaCorrection<<<gridSize, blockSize>>>(gpuData,gamma,width*height*channels);
    auto t_endGPU = std::chrono::high_resolution_clock::now();

    HIP_ASSERT(hipDeviceSynchronize());
    HIP_ASSERT(hipMemcpy(cpuData, gpuData, imageDataSize, hipMemcpyDeviceToHost));

    Mat gpuProcessedImage(height, width, CV_8UC3, cpuData);
    imwrite(outputFile, gpuProcessedImage);

    double elapsed_time_ns_GPU = std::chrono::duration<double, std::nano>(t_endGPU-t_startGPU).count();

    cout<<"GPU time: "<<elapsed_time_ns_GPU<<" ns\n";
}   

int main(){
    for(int i=0;i<2048;i+=256){
        int gridSize = i;
        int blockSize = 256;
        cout<<"\n-------------------\nGrid size: "<<gridSize;
        test(gridSize, blockSize);
    }
}