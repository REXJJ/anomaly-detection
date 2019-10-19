#ifndef REFERENCE_BS_HPP
#define REFERENCE_BS_HPP
#include <iostream>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include "Debugging_Utilities/debugging_utilities.h"
  
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
    MatrixXf mean_A;
    MatrixXf std_A;
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
      X(i,j)=float(xb-xa);
      Y(i,j)=float(yb-ya);   
      M(i,j)=sqrt(X(i,j)*X(i,j)+Y(i,j)*Y(i,j));   
      A(i,j)=atan2(Y(i,j),X(i,j));
    }
  }
}

void image_params::stat_calculation()
{
  for(int i=0;i<image.rows;i++)
  {
    for(int j=0;j<image.cols;j++)
    {
      if(i-kernel_size/2<0||j-kernel_size/2<0||i+kernel_size/2>=image.rows||j+kernel_size/2>=image.cols)
      {
        mean_m(i,j)=0.0;
        std_m(i,j)=0.0;        
        continue;
      }
      MatrixXf sub=M.block(i-kernel_size/2,j-kernel_size/2,kernel_size,kernel_size); 
      mean_m(i,j)=sub.mean();
      MatrixXf temp=sub-MatrixXf::Ones(sub.rows(),sub.cols())*mean_m(i,j);
      MatrixXf squares=temp.array()*temp.array();
      // for(int a=0;a<temp.rows();a++)
      // {
      // 	for(int b=0;b<temp.cols();b++)
      // 	{
      // 		std::cout<<temp(a,b)<<" ";
      // 	}
      // 	std::cout<<endl;
      // }
      // std::cout<<"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
      std_m(i,j)=sqrt(squares.sum())/kernel_size;
      MatrixXf sub_A=A.block(i-kernel_size/2,j-kernel_size/2,kernel_size,kernel_size);
      mean_A(i,j)=sub_A.mean();
      MatrixXf temp_A=sub_A-MatrixXf::Ones(sub.rows(),sub.cols())*mean_A(i,j);
      MatrixXf squares_A=temp_A.array()*temp_A.array();
      std_A(i,j)=sqrt(squares_A.sum())/kernel_size;
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
  mean_A.resize(img.rows,img.cols);
  std_A.resize(img.rows,img.cols);
  K=k;
  kernel_size=ks;
  gradient_calculation();
}

Mat get_mask(MatrixXf& ref_mean,MatrixXf& test_mean, MatrixXf& ref_sd,MatrixXf& test_sd,MatrixXf& refA_mean,MatrixXf& testA_mean)
{
	Mat mask(cv::Size(ref_mean.cols(),ref_mean.rows()),CV_8UC1,Scalar(0));
	for(int i=0;i<ref_mean.rows();i++)
	{
		for(int j=0;j<ref_mean.cols();j++)
		{
			if(abs(ref_mean(i,j)-test_mean(i,j))>36||abs(ref_sd(i,j)-test_mean(i,j))>100||abs(refA_mean(i,j)-testA_mean(i,j))>55)
				mask.at<uchar>(i,j)=255;
		}
	}
	return mask;
}

#endif
