#ifndef REFERENCE_BS_HPP
#define REFERENCE_BS_HPP
#include <iostream>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <unordered_set>
#include <fstream>
#include "/home/rex/Desktop/REX_WS/AnomalousRegion1/src/Debugging_Utilities/debugging_utilities.h"  
using namespace std;
using namespace cv;
using namespace Eigen;

class image_params
{
  public:
    Mat image;
    MatrixXf X;//X Gradient
    MatrixXf Y;//Y Gradient
    MatrixXf M;//Magnitude of the gradients
    MatrixXf A;//Angle between the x and the y gradient.
    MatrixXf mean_m;//Mean of the magnitude.
    MatrixXf std_m;//Standard Deviation of the magnitude.
    MatrixXf mean_A;//Mean of the Angle.
    MatrixXf std_A;//Standard Deviation of the Angle.
    VectorXi per;//Stores the computed percentile of the gradients.
    vector<pair<int,int>> filtered;//
    MatrixXi ids;//

    int K,kernel_size;//K is the number of rows or columns taken for calculating gradients. Kernel_size is the size of the sliding window for statistical calculation.
    image_params(Mat& img, int k,int ks);
    void gradient_calculation();
    void stat_calculation();
    void percentile();
    Mat filter_vectors(double p_min,double p_max);
};

void image_params::gradient_calculation()
{
  for(size_t i=0;i<image.rows;i++)
  {
    for(size_t j=0;j<image.cols;j++)
    {
      if(i-K<0||j-K<0||i+K>=image.rows||j+K>image.cols||j<60||i<102||i>573)
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
      M(i,j)=abs(sqrt(X(i,j)*X(i,j)+Y(i,j)*Y(i,j)));   
      if(M(i,j)!=M(i,j)) M(i,j)=0;
      A(i,j)=atan2(Y(i,j),X(i,j));
      if(A(i,j)!=A(i,j)) A(i,j)=0;
      int index=int(M(i,j));
      per(index)=per(index)+1;
    }
  }
}

void image_params::stat_calculation()//The time consuming function. 
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

void image_params::percentile()
{
  for(int i=1;i<360;i++){
    per(i)=per(i)+per(i-1);
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
  ids=MatrixXi::Ones(img.rows,img.cols)*-1;
  K=k;
  kernel_size=ks;
  per=VectorXi::Zero(360);
  gradient_calculation();
}

typedef MatrixXf Mx;

Mat get_mask(Mx& ref_mean_mag,Mx& test_mean_mag, Mx& ref_sd_mag,Mx& test_sd_mag,Mx& ref_mean_ang,Mx& test_mean_ang,Mx& ref_sd_ang, Mx& test_sd_ang,vector<double> &params)
{
	int r=ref_mean_mag.rows(),c=ref_mean_mag.cols();
	Mat mask(cv::Size(c,r),CV_8UC1,Scalar(0));
	for(int i=0;i<r;i++)
	{
		for(int j=0;j<c;j++)
		{
			// if(abs(ref_mean(i,j)-test_mean(i,j))>10||abs(ref_sd(i,j)-test_sd(i,j))>10||abs(refA_mean(i,j)-testA_mean(i,j))>55)
      if(abs(ref_mean_mag(i,j)-test_mean_mag(i,j))>params[0]&&abs(ref_sd_mag(i,j)-test_sd_mag(i,j))>params[1])
				mask.at<uchar>(i,j)=255;
		}
	}
	return mask;
}

unordered_set<int> get_mask_set(Mx& ref_mean_mag,Mx& test_mean_mag, Mx& ref_sd_mag,Mx& test_sd_mag,Mx& ref_mean_ang,Mx& test_mean_ang,Mx& ref_sd_ang, Mx& test_sd_ang,vector<double> &params)
{
  unordered_set<int> mask_points;
  int r=ref_mean_mag.rows(),c=ref_mean_mag.cols();
  Mat mask(cv::Size(c,r),CV_8UC1,Scalar(0));
  for(int i=0;i<r;i++)
  {
    for(int j=0;j<c;j++)
    {
      // if(abs(ref_mean(i,j)-test_mean(i,j))>10||abs(ref_sd(i,j)-test_sd(i,j))>10||abs(refA_mean(i,j)-testA_mean(i,j))>55)
      if(abs(ref_mean_mag(i,j)-test_mean_mag(i,j))>params[0]&&abs(ref_sd_mag(i,j)-test_sd_mag(i,j))>params[1])
        mask_points.insert(10000*i+j);
    }
  }
  return mask_points;
}

Mat image_params::filter_vectors(double p_min,double p_max)
{
    Mat mask(cv::Size(M.cols(),M.rows()),CV_8UC1,Scalar(0));
    int count=0;
    for(int i=0;i<M.rows();i++)
      for(int j=0;j<M.cols();j++)
      {
        if(M(i,j)!=M(i,j)||(X(i,j)==0&&Y(i,j)==0)) M(i,j)=0;
        int index=abs(int(M(i,j)));
        // std::cout<<M(i,j)<<" "<<X(i,j)<<" "<<Y(i,j)<<" "<<A(i,j)<<endl;
        // if(X(i,j)==0&&Y(i,j)==0) std::cout<<M(i,j)<<" "<<A(i,j)<<endl;
        double p = double(per(index))/double(per(359))*100;
        if(p>=p_min&&p<=p_max){
          mask.at<uchar>(i,j)=255;
          filtered.push_back(make_pair(i,j));
          ids(i,j)=count;
          count++;
        }
      }
      return mask;
}

#endif
