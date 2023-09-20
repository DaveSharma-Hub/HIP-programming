// Online C++ compiler to run C++ program online
#include <iostream>
#include <vector>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

class Pixel {
public:
    int red;
    int green;
    int blue;
    Pixel(int red, int green, int blue) {
        this->red = red;
        this->green = green;
        this->blue = blue;
    }
};

class Converter {
public:
    vector<vector<Pixel*>> pixelGrid;
    Converter(vector<vector<Pixel*>>& grid) {
        copyPixelGrid(grid);
    }

    void toYUV(int first, int second, int third) {
        //eg: 4,4,4 etc..
        for (int heightB = 0, heightE = heightB + 1; heightE < this->pixelGrid.size(); heightB += 2, heightE = heightB + 1) {
            for (int beg = 0, end = beg + 3; end < this->pixelGrid[heightB].size(); beg += 4, end = beg + 3) {
                //first row
                Pixel* firstOne = this->pixelGrid[heightB][beg];
                Pixel* firstTwo = this->pixelGrid[heightB][beg + 1];
                Pixel* firstThree = this->pixelGrid[heightB][beg + 2];
                Pixel* firstFour = this->pixelGrid[heightB][beg + 3];
                // second row
                Pixel* secondOne = this->pixelGrid[heightE][beg];
                Pixel* secondTwo = this->pixelGrid[heightE][beg + 1];
                Pixel* secondThree = this->pixelGrid[heightE][beg + 2];
                Pixel* secondFour = this->pixelGrid[heightE][beg + 3];
                //sampling first row
                if (second == 4) {
                    this->pixelGrid[heightB][beg] = firstOne;
                    this->pixelGrid[heightB][beg + 1] = firstTwo;
                    this->pixelGrid[heightB][beg + 2] = firstThree;
                    this->pixelGrid[heightB][beg + 3] = firstFour;
                }
                else if (second == 2) {
                    this->pixelGrid[heightB][beg] = firstOne;
                    this->pixelGrid[heightB][beg + 1] = firstOne;
                    this->pixelGrid[heightB][beg + 2] = firstThree;
                    this->pixelGrid[heightB][beg + 3] = firstThree;
                }
                else if (second == 1) {
                    this->pixelGrid[heightB][beg] = firstOne;
                    this->pixelGrid[heightB][beg + 1] = firstOne;
                    this->pixelGrid[heightB][beg + 2] = firstOne;
                    this->pixelGrid[heightB][beg + 3] = firstOne;
                }
                //sampling second row
                if (third == 4) {
                    this->pixelGrid[heightE][beg] = secondOne;
                    this->pixelGrid[heightE][beg + 1] = secondTwo;
                    this->pixelGrid[heightE][beg + 2] = secondThree;
                    this->pixelGrid[heightE][beg + 3] = secondFour;
                }
                else if (third == 2) {
                    this->pixelGrid[heightE][beg] = secondOne;
                    this->pixelGrid[heightE][beg + 1] = secondOne;
                    this->pixelGrid[heightE][beg + 2] = secondThree;
                    this->pixelGrid[heightE][beg + 3] = secondThree;
                }
                else if (third == 1) {
                    this->pixelGrid[heightE][beg] = secondOne;
                    this->pixelGrid[heightE][beg + 1] = secondOne;
                    this->pixelGrid[heightE][beg + 2] = secondOne;
                    this->pixelGrid[heightE][beg + 3] = secondOne;
                }
            }
        }
    }

    void copyPixelGrid(vector<vector<Pixel*>>& grid) {
        for (int i = 0; i < grid.size(); i++) {
            vector<Pixel*> tmpGrid;
            for (Pixel* p : grid[i]) {
                tmpGrid.push_back(p);
            }
            this->pixelGrid.push_back(tmpGrid);
        }
    }

    void print() {
        for (int i = 0; i < this->pixelGrid.size(); i++) {
            for (Pixel* p : this->pixelGrid[i]) {
                cout << p->red << ":" << p->green << ":" << p->blue << " ";
            }
            cout << "\n";
        }
    }
};

int randomRGBValue() {
    return rand() % 256;
}

vector<vector<Pixel*>> init(int row, int column) {
    vector<vector<Pixel*>> grid;

    for (int i = 0; i < row; i++) {
        vector<Pixel*> tmp;
        for (int j = 0; j < column; j++) {
            int r = randomRGBValue();
            int g = randomRGBValue();
            int b = randomRGBValue();
            Pixel* p = new Pixel(r, g, b);
            tmp.push_back(p);
        }
        grid.push_back(tmp);
    }
    return grid;
}

vector<vector<Pixel*>> convertImageToPixel(string input) {
    vector<vector<Pixel*>> grid;

    Mat foo = imread(input);
    uint8_t* pixelPtr = (uint8_t*)foo.data;
    int cn = foo.channels();

    for (int i = 0; i < foo.rows; i++)
    {
        vector<Pixel*> tmp;
        for (int j = 0; j < foo.cols; j++)
        {
            int b = (int)pixelPtr[i * foo.cols * cn + j * cn + 0]; // B
            int g = (int)pixelPtr[i * foo.cols * cn + j * cn + 1]; // G
            int r = (int)pixelPtr[i * foo.cols * cn + j * cn + 2]; // R
            Pixel* p = new Pixel(r, g, b);
            tmp.push_back(p);
        }
        grid.push_back(tmp);
    }

    return grid;
}

void convertPixelToImage(vector<vector<Pixel*>>& grid, string output) {

    for (int i = 0; i < grid.size(); i++) {
        for (int j = 0; j < grid[i].size(); j++) {
            // add the pixels?
        }
    }
}

int main() {
    // Write C++ code here

    // vector<vector<Pixel*>> grid = init(10,10);
    string image = "./image.jpg";
    vector<vector<Pixel*>> grid = convertImageToPixel(image);

    Converter* c1 = new Converter(grid);
    // c1->print();
    c1->toYUV(4, 2, 2);
    cout << "-------------------------" << endl;
    // c1->print();
    convertPixelToImage(c1->pixelGrid, image);
    std::cout << "Hello world!";

    return 0;
}