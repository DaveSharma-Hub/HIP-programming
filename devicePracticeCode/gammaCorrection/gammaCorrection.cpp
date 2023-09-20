#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

using namespace std;
using namespace cv;

__global__ void gammaCorrection(uchar* image, float gamma, int num_values){
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(id<num_values){
        image[id] = pow((uint)image[id]/255.0, gamma) * 255.0;
    }
}

void test(string imagePath, string outputFile, float gamma){
    Mat image = imread(imagePath, IMREAD_COLOR);

    int width = image.cols;
    int height = image.rows;
    int channels = 3; 
    int num_values = width * height * channels;

    // float gamma = 4.0;

    int blockSize = 256; //threds per block
    int gridSize = (num_values + blockSize - 1) / blockSize;    

    uchar* imageData;// get image data
    int imageDataSize = sizeof(uchar) * width * height * channels;
    imageData = (uchar*)malloc(imageDataSize);

    memcpy(imageData, image.data, imageDataSize);

    uchar* gpuData;
    uchar* resultData = (uchar*)malloc(imageDataSize);

    HIP_ASSERT(hipMalloc(&gpuData,imageDataSize));

    HIP_ASSERT(hipMemcpy(gpuData, imageData,imageDataSize ,hipMemcpyHostToDevice));

    gammaCorrection<<<gridSize, blockSize>>>(gpuData,gamma,num_values);
    HIP_ASSERT(hipDeviceSynchronize());

    HIP_ASSERT(hipMemcpy(resultData,gpuData,imageDataSize,hipMemcpyDeviceToHost));

    cv::Mat gpuProcessedImage(image.rows, image.cols, CV_8UC3, resultData);
    cv::imwrite(outputFile, gpuProcessedImage);

    HIP_ASSERT(hipFree(gpuData));
    free(imageData);
    free(resultData);
}

int main(){
    string inputFile = "./image2.jpg";
    for(int i=0;i<10;i++){
        string outputFile = "output" + to_string(i) + ".jpg"; 
        test(inputFile, outputFile, (float)i);
    }
}