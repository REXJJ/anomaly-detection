#include <iostream>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <fstream>
using Eigen::MatrixXd;
 
using namespace std;
using namespace cv;
using namespace Eigen;

class image_params
{
  public:
    Mat image;
    MatrixXf X;
    MatrixXf Y;
    MatrixXf M;
    MatrixXf A;
    MatrixXf mean_m;
    MatrixXf std_m;
    int K,kernel_size;
    image_params(Mat& img, int k,int ks);
    void gradient_calculation();
    void stat_calculation();
};

void image_params::gradient_calculation()
{
  for(size_t i=0;i<image.rows;i++)
  {
    for(size_t j=0;j<image.cols;j++)
    {
      if(i-K<0||j-K<0||i+K>=image.rows||j+K>image.cols)
      {
        X(i,j)=0.0;
        Y(i,j)=0.0;
        continue;
      }
      int xa=0,xb=0,ya=0,yb=0,count=0;
      for(int u=-K;u<=K;u++)
      {
        for(int v=1;v<=K;v++)
        {
          ya+=int(image.at<uchar>(i-v,j+u));
          yb+=int(image.at<uchar>(i+v,j+u));
          xa+=int(image.at<uchar>(i+u,j-v));
          xb+=int(image.at<uchar>(i+u,j+v));
          count++;
        }
      }
      X(i,j)=float(xb-xa)/count;
      Y(i,j)=float(yb-ya)/count;   
      M(i,j)=sqrt(X(i,j)*X(i,j)+Y(i,j)*Y(i,j));   
      A(i,j)=atan2(Y(i,j),X(i,j));
    }
  }
}

void image_params::stat_calculation()
{
  for(size_t i=0;i<image.rows;i++)
  {
    for(size_t j=0;j<image.cols;j++)
    {
      if(i-K<0||j-K<0||i+K>=image.rows||j+K>image.cols)
      {
        X(i,j)=0.0;
        Y(i,j)=0.0;
        continue;
      }
      //mean_m(i,j)=mean of the block
      //std_m(i,j)=standard deviation of the block
    }
  }
}

image_params::image_params(Mat& img,int k,int ks)
{
  image=img;
  X.resize(img.rows,img.cols);
  Y.resize(img.rows,img.cols);
  M.resize(img.rows,img.cols);
  A.resize(img.rows,img.cols);
  mean_m.resize(img.rows,img.cols);
  std_m.resize(img.rows,img.cols);
  K=k;
  kernel_size=ks;
  gradient_calculation();
}

int main( int argc, char** argv)
{

  ofstream f1,f2;
  f1.open("x_test.csv",ios::out);
  f2.open("y_test.csv",ios::out);
 
  Mat reference,test;
  if( argc < 2)
    {
     cout <<" Usage: ReferenceImage TestImage" << endl;
     return -1;
    }

  Mat out(reference.size(),reference.type(),0);

  reference=cv::imread(argv[1],0);
  std::cout<<reference.rows<<endl;
  test=cv::imread(argv[2],0);
  image_params image(reference,2,7);
  for(size_t i=0;i<image.X.rows();i++){
     for(size_t j=0;j<image.X.cols();j++){
       f1<<image.X(i,j)<<", ";
     }
     f1<<"\n";
   }

  for(size_t i=0;i<image.Y.rows();i++){
     for(size_t j=0;j<image.Y.cols();j++){
       f2<<image.Y(i,j)<<", ";
     }
     f2<<"\n";
   }
  return 0;
}