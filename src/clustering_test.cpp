#include "cv.h"
#include "highgui.h"
#include "include/reference_bs.hpp"
#include "include/clustering_functions.hpp"


using namespace cv;
using namespace std;

RNG rng(12345);


int main( int argc, char** argv )
{
  VideoCapture cap("/home/rex/Desktop/ICRA/test1.mp4"); 
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  Rect ROI(704,243,723,650);  
  Mat reference,test;
  reference=imread("/home/rex/Desktop/ICRA/ref_image.jpg",0);
  image_params image(reference,2,7);
  // image.stat_calculation();
  int count=0;
  while(1){ 
      Mat frame;
      cap >> frame;
      if (frame.empty())
        break; 
      frame=frame(ROI);
      cvtColor(frame,test, CV_BGR2GRAY);

      // if( argc < 2)
      //   {
      //    cout <<" Usage: ReferenceImage TestImage" << endl;
      //    return -1;
      //   }

      image_params test_(test,2,7);
      // // test_.stat_calculation();
      test_.percentile();
      Mat mask=test_.filter_vectors();
      MatrixXi mk=MatrixXi::Zero(mask.rows,mask.cols);
      for(int i=0;i<mk.rows();i++)
      {
        for(int j=0;j<mk.cols();j++)
        {
          if(mask.at<uchar>(i,j)==255)
            mk(i,j)=1;
        }
      }
      int si=0;
      Mat drawing = Mat::zeros(mask.size(), CV_8UC3 );
      clustering cl(test_.A,test_.M,mk,test_.filtered,test_.ids);
      VectorXi classes=cl.cluster();
      vector<unordered_set<int>> clusters=cl.clusters;

      for(auto x:clusters)
      {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        vector<Point2f> v;
        for(auto y:x)
        {
          v.push_back(Point2f(double(test_.filtered[y].second),double(test_.filtered[y].first)));
        }
        RotatedRect r=minAreaRect(v);
        Point2f rect_points[4]; r.points( rect_points );
        for( int j = 0; j < 4; j++ )
          line( frame, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
      }

     Mat new_mask(cv::Size(mk.cols(),mk.rows()),CV_8UC1,Scalar(0));
     for(int i=0;i<classes.size();i++)
      {
        if(classes(i)!=0)
        {
          new_mask.at<uchar>(test_.filtered[i].first,test_.filtered[i].second)=255;
        }
      }
      // for(int i=100;i<mk.rows();i++)
      // {
      //   for(int j=100;j<mk.cols();j++)
      //   {
      //     vector<pair<int,int>> t=cl.find_neighbors(i,j,test_.A(i,j),test_.M(i,j));
      //     si=t.size();
      //     if(t.size()>200){
      //       for(auto pts:t)
      //       {
      //         new_mask.at<uchar>(pts.first,pts.second)=255;
      //       }
      //       goto out;
      //     }
      //   }
      // }
      // std::cout<<"Not Found"<<endl;
      // out:
      // std::cout<<"Size of Cluster: "<<si<<endl;
      namedWindow("Before Filtering", WINDOW_AUTOSIZE );
      imshow( "Before Filtering",mask ); 
      namedWindow( "Display window", WINDOW_AUTOSIZE );
      imshow( "Display window", new_mask );     
      namedWindow( "Real Image", WINDOW_AUTOSIZE );
      imshow( "Real Image", test );   
      // namedWindow( "Clusters", WINDOW_AUTOSIZE );
      // imshow( "Clusters", drawing );            
      std::cout<<count++<<endl;
      imshow( "Frame", frame);
      char c=(char)waitKey(25);
      if(c==27)
          break;
  }
  cap.release();
  destroyAllWindows();
  return 0;
}