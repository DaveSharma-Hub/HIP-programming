// Online C++ compiler to run C++ program online
#include <iostream>
#include <vector>
#include <cstdlib>
#include <windows.h>
#include <string.h>

using namespace std;

class Pixel{
    public:
        int red;
        int green;
        int blue;
        Pixel(int red, int green, int blue){
            this->red = red;
            this->green = green;
            this->blue = blue;
        }
};

class ImageScale{
    public:
        vector<vector<Pixel*>> oldGrid;
        vector<vector<Pixel*>> scaledGrid;
        
        ImageScale(vector<vector<Pixel*>>& grid){
            copyGrid(grid);
        }
        
        void copyGrid(vector<vector<Pixel*>>& grid){
            
            for(int i=0;i<grid.size();i++){
                vector<Pixel*> tmp;
                for(int j=0;j<grid[i].size();j++){
                    tmp.push_back(grid[i][j]);
                }
                this->oldGrid.push_back(tmp);
            }
        }
        
        void scale(int rowFactor, int columnFactor){
            for(int i=0;i<this->oldGrid.size();i++){
                vector<Pixel*> tmpRow;
                for(int j=0;j<this->oldGrid[i].size();j++){
                    Pixel* value = this->oldGrid[i][j];
                    for(int colCount=0;colCount<columnFactor;colCount++){
                        tmpRow.push_back(value);
                    }
                }
                this->scaledGrid.push_back(tmpRow);
                for(int rowCount=0;rowCount<rowFactor;rowCount++){
                    this->scaledGrid.push_back(tmpRow);
                }
                scaledGrid.push_back(tmpRow);
            }
        }
        
        
        void printNew(){
            for(int i=0;i<scaledGrid.size();i++){
                for(int j=0;j<scaledGrid[i].size();j++){
                    Pixel* p = scaledGrid[i][j];
                    cout<<p->red<<":"<<p->green<<":"<<p->blue<<" ";
                }
                cout<<"\n";
            }
        }
        void printOld(){
            for(int i=0;i<oldGrid.size();i++){
                for(int j=0;j<oldGrid[i].size();j++){
                    Pixel* p = oldGrid[i][j];
                    cout<<p->red<<":"<<p->green<<":"<<p->blue<<" ";
                }
                cout<<"\n";
            }
        }
};

int randomRGB(){
    return rand()%256;
}

vector<vector<Pixel*>> init(int row, int col){
    vector<vector<Pixel*>> grid;
    for(int i=0;i<row;i++){
        vector<Pixel*> tmp;
        for(int j=0;j<col;j++){
            Pixel* p = new Pixel(randomRGB(),randomRGB(),randomRGB());
            tmp.push_back(p);
        }
        grid.push_back(tmp);
    }
    return grid;
}


vector<vector<Pixel*>> convertImageToPixel(string input){
    vector<vector<Pixel*>> grid;
    for(int i=0;i<;i++){
        vector<Pixel*> tmp;
        for(int j=0;j<;j++){
            int r = ;
            int g = ;
            int b = ;
            Pixel* p = new Pixel(r,g,b);
            tmp.push_back(p);
        }
        grid.push_back(tmp);
    }
    return grid;
}

void convertPixelToImage(vector<vector<Pixel*>>& grid, string output){
    
    for(int i=0;i<grid.size();i++){
        for(int j=0;j<grid[i].size();j++){
            // add the pixels?
        }
    }
}



int main() {
    // Write C++ code here
    
    // vector<vector<Pixel*>> og = init(10,10);

    string image = "./image.jpg";
    string newImage = "./newImage.jpg";
    vector<vector<Pixel*>> imagePixels = convertImageToPixel(image);
    ImageScale* scaled = new ImageScale(imagePixels); 
    scaled->printOld();
    scaled->scale(2,3);
    cout<<"--------------------\n";
    scaled->printNew();
    convertPixelToImage(scaled->scaledGrid,newImage);
    std::cout << "Completed!";
    return 0;
}