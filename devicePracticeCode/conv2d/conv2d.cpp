#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include "../gamma/gammaCorrection.hpp"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

using namespace std;
using namespace cv;
using namespace GAMMA;

__global__ void conv2d(uchar* imageData, float* mask, int imageWidth, int imageHeight, int maskWidth, int maskHeight, int channels){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x >= imageWidth || y >= imageHeight){
        return;
    }

    float sumR = 0;
    float sumG = 0;
    float sumB = 0;
    for(int i=0;i<maskWidth;i++){
        for(int j=0;j<maskHeight;j++){
            int imageX = x + i - maskWidth/2;
            int imageY = y + j - maskHeight/2;
            if(imageX<0 || imageX >= imageWidth || imageY < 0 || imageY >= imageHeight){
                continue;
            }
            int imageIndexR = (channels * imageY * imageWidth) + channels* imageX;
            int imageIndexG = (channels * imageY * imageWidth) + channels* imageX + 1;
            int imageIndexB = (channels * imageY * imageWidth) + channels* imageX + 2;
            int maskIndex = j * maskWidth + i;
            sumR += imageData[imageIndexR] / 255.0f * mask[maskIndex];
            sumG += imageData[imageIndexG] / 255.0f * mask[maskIndex];
            sumB += imageData[imageIndexB] / 255.0f * mask[maskIndex];
        }
    }

    int imageIndexR = (channels * y * imageWidth) + channels * x;
    int imageIndexG = (channels * y * imageWidth) + channels * x + 1;
    int imageIndexB = (channels * y * imageWidth) + channels * x + 2;
    imageData[imageIndexR] = sumR * 255;
    imageData[imageIndexG] = sumG * 255;
    imageData[imageIndexB] = sumB * 255;
}

void test(int maskWidth, int maskHeight){
    string imagePath = "./image2.jpg";
    string outputFile = "output_brightened"+ to_string(maskWidth) + ".jpg";

    Mat image = imread(imagePath,IMREAD_COLOR);
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    int imageDataSize = width*height*channels*sizeof(uchar);
    uchar* imageData;
    imageData = (uchar*)malloc(imageDataSize);
    memcpy(imageData, image.data, imageDataSize);

    // static const int maskWidth = 200;
    // static const int maskHeight = 200;

    vector<float> mask(maskWidth * maskHeight * channels);
    for(int i=0;i<maskWidth * maskHeight;++i){
        mask[i] = 1.0f / (maskWidth * maskHeight * channels);
    }

    uchar* resultData = (uchar*)malloc(imageDataSize);
    uchar* gpuData;
    float* dMask;

    HIP_ASSERT(hipMalloc(&gpuData,imageDataSize));
    HIP_ASSERT(hipMalloc(&dMask, maskWidth * maskHeight * channels * sizeof(float)));

    HIP_ASSERT(hipMemcpy(gpuData, imageData, imageDataSize, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(dMask, mask.data(), maskWidth * maskHeight * channels * sizeof(float), hipMemcpyHostToDevice));

    dim3 blockSize = {16,16,1};
    dim3 gridSize = {(width + blockSize.x -1)/blockSize.x, (height + blockSize.y -1)/blockSize.y, 1};

    conv2d<<<gridSize, blockSize>>>(gpuData,dMask, width, height, maskWidth, maskHeight, channels);

    HIP_ASSERT(hipDeviceSynchronize());
    
    HIP_ASSERT(hipMemcpy(resultData, gpuData, imageDataSize, hipMemcpyDeviceToHost));

    uchar* brightenedImage = gammaFunction(resultData, width, height, channels, 2.0);

    Mat gpuProcessedImage(image.rows, image.cols, CV_8UC3, brightenedImage);
    imwrite(outputFile, gpuProcessedImage);

    HIP_ASSERT(hipFree(gpuData));
    HIP_ASSERT(hipFree(dMask));

    free(resultData);
    free(imageData);
}


int main(){
    // vector<int> sizes = {25,50,100,200, 400, 600 ,800};
    // for(int size : sizes){
    //     cout<<"\n-------------\nMask size: "<<size<<endl;
    //     test(size, size);
    // }
    test(50,50);
}  