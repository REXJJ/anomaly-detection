#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <unordered_set>
#include <chrono>
#include "cv.h"
#include "highgui.h"
#include "include/reference_bs.hpp"
#include "include/clustering_functions.hpp"


using namespace cv;
using namespace std;

RNG rng(12345);

int max_value = 100;
int low=90;
int high = 95;
string window_detection_name;

static void on_low(int, void *)
{
    low = min(high-1, low);
    setTrackbarPos("Low", window_detection_name, low);
}
static void on_high(int, void *)
{
    high = max(high, low+1);
    setTrackbarPos("High", window_detection_name, high);
}


int mag_mean=0;
int mag_sd=0;

static void mean_magnitude(int, void *)
{
    setTrackbarPos("magnitude_mean", window_detection_name, mag_mean);
}
static void sd_magnitude(int, void *)
{
    setTrackbarPos("magnitude_sd", window_detection_name, mag_sd);
}




bool lines_intersect(double l1[2][2], double l2[2][2])
{
	// l1 for horizontal ray line...slope is always zero

	// checking if other slope is zero
	if (l2[0][1]==l2[1][1])
	{
    	return false;
	}
	else
	{
		// checking both pts of second line above first line
		if ((l2[0][1]>l1[0][1] && l2[1][1]>l1[0][1]) || (l2[0][1]<l1[0][1] && l2[1][1]<l1[0][1]))
		{
			return false;
		}
		else
		{
			// checking both pts of second line either on right or on left of fist line
			if ((l2[0][0]<l1[0][0] && l2[1][0]<l1[0][0]) || (l2[0][0]>l1[1][0] && l2[1][0]>l1[1][0]))
			{
				return false;
			}
			else
			{
				// checking if other line is vertical
				if (l2[0][0]== l2[1][0])
				{
					return true;
				}
				else
				{
					// getting intersection point
					double m2 = (l2[1][1]-l2[0][1])/(l2[1][0]-l2[0][0]);		
					double x = (l1[0][1]+m2*l2[0][0]-l2[0][1])/m2;
					// checking if intersection point lies on the first line
					if ((x>l1[0][0] || std::abs(x-l1[0][0])<1e-9) && (x<l1[1][0] || std::abs(x-l1[1][0])<1e-9))
					{
						return true;
					}
					else
					{
						return false;
					}
				}
			}
		}
	} 
	return false;
}

void InPoly(const Eigen::MatrixXd& q, const Eigen::MatrixXd& p, Eigen::MatrixXd& in)
{
    double l1[2][2];
    double l2[2][2];

    double xmin = p.col(0).minCoeff();
    double xmax = p.col(0).maxCoeff();
    double ymin = p.col(1).minCoeff();
    double ymax = p.col(1).maxCoeff();

    for (long i=0;i<q.rows();++i)
    {
        // bounding box test
        if (q(i,0)<xmin || q(i,0)>xmax || q(i,1)<ymin || q(i,1)>ymax)
        {
            continue;
        }
        int intersection_count = 0;
        Eigen::MatrixXd cont_lines = Eigen::MatrixXd::Constant(p.rows(),1,0);
        for (int j=0;j<p.rows();++j)
        {
            if (j==0)
            {
                l1[0][0] = q(i,0);l1[0][1] = q(i,1);
                l1[1][0] = xmax;l1[1][1] = q(i,1);
                l2[0][0] = p(p.rows()-1,0);l2[0][1] = p(p.rows()-1,1);
                l2[1][0] = p(j,0);l2[1][1] = p(j,1);
                if (lines_intersect(l1,l2))
                {
                    intersection_count++;
                    cont_lines(j,0) = 1;
                }   
            }
            else
            {
                l1[0][0] = q(i,0);l1[0][1] = q(i,1);
                l1[1][0] = xmax;l1[1][1] = q(i,1);
                l2[0][0] = p(j,0);l2[0][1] = p(j,1);
                l2[1][0] = p(j-1,0);l2[1][1] = p(j-1,1);
                if (lines_intersect(l1,l2))
                {
                    intersection_count++;
                    cont_lines(j,0) = 1;
                    if (cont_lines(j-1,0)==1)
                    {
                        if (p(j-1,1)==q(i,1))
                        {
                            if (j-1==0)
                            {
                                if (!((p(p.rows()-1,1)<p(j-1,1) && p(j,1)<p(j-1,1)) || (p(p.rows()-1,1)>p(j-1,1) && p(j,1)>p(j-1,1))))
                                {
                                    intersection_count--;
                                }
                            }
                            else
                            {
                                if (!((p(j-2,1)<p(j-1,1) && p(j,1)<p(j-1,1)) || (p(j-2,1)>p(j-1,1) && p(j,1)>p(j-1,1))))
                                {
                                    intersection_count--;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (intersection_count%2==1)
        {
            in(i,0) = 1;
        }
        else
        {
        	in(i,0) = 0;
        }
    }
}


int main( int argc, char** argv )
{
  window_detection_name="Control";
  namedWindow("Before Filtering", WINDOW_AUTOSIZE );
  namedWindow( "Display window", WINDOW_AUTOSIZE );
  namedWindow( "Real Image", WINDOW_AUTOSIZE );
  namedWindow(window_detection_name,WINDOW_AUTOSIZE);

  createTrackbar("Low", window_detection_name, &low, max_value, on_low);
  createTrackbar("High", window_detection_name, &high, max_value, on_high);

  createTrackbar("magnitude_mean", window_detection_name, &mag_mean,100,mean_magnitude);
  createTrackbar("magnitude_sd", window_detection_name, &mag_sd,100,sd_magnitude);

  // VideoCapture cap("/home/rex/Desktop/ICRA/test1.mp4"); 
  VideoCapture cap(0); 
  cap.set(CV_CAP_PROP_AUTO_EXPOSURE, 0.25);
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  Rect ROI(167,61,400,360);  
  Mat reference,test;
  reference=imread("/home/cl/Desktop/REX_WS/AnomalousRegion1/Images/ref_image1.jpg",0);
  image_params image(reference,2,9);
  // image.stat_calculation();
  typedef boost::geometry::model::d2::point_xy<double> point_type;
  typedef boost::geometry::model::polygon<point_type> polygon_type;
  polygon_type poly;
  boost::geometry::read_wkt(
  "POLYGON((32 43,27 66,42 141,68 243,76 301,95 355,353 353,360 324,373 219,381 116,385 70,371 30,367 10,291 11,170 17,164 23,64 38))", poly);
  Eigen::MatrixXd points(17,2);
  points << 32,43,27,66,42, 141,68, 243,76, 301,95, 355,353, 353,360, 324,373, 219,381, 116,385, 70,371, 30,367, 10,291, 11,170, 17,164, 23,64, 38;
  
  int count=0;
  while(1){ 
      Mat frame;
      cap >> frame;
      if (frame.empty())
        break; 
      // frame=frame(ROI);
      cvtColor(frame,test, CV_BGR2GRAY);

      // if( argc < 2)
      //   {
      //    cout <<" Usage: ReferenceImage TestImage" << endl;
      //    return -1;
      //   }

      image_params test_(test,2,9);
      // // test_.stat_calculation();

      test_.stat_calculation();
      vector<double> params={double(mag_mean),double(mag_sd),20,10};
      unordered_set<int> mask_set=get_mask_set(image.mean_m,test_.mean_m,image.std_m,test_.std_m,image.mean_A,test_.mean_A,image.std_A,test_.std_A,params);
      Mat dummy=get_mask(image.mean_m,test_.mean_m,image.std_m,test_.std_m,image.mean_A,test_.mean_A,image.std_A,test_.std_A,params);

      test_.percentile();

      Mat mask=test_.filter_vectors(low,high);
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
        int count=0;
        bool status=false;
        for(auto y:x)
        {
          v.push_back(Point2f(double(test_.filtered[y].second),double(test_.filtered[y].first)));
          if(mask_set.find(test_.filtered[y].first*10000+test_.filtered[y].second)!=mask_set.end()) count++;
          point_type p(test_.filtered[y].first,test_.filtered[y].second);
          // if(!boost::geometry::within(p, poly)) status=true;
          auto clock_start = std::chrono::high_resolution_clock::now();
          for(int kl=0;kl<100000;kl++)
          {
	          point_type p_t(rand()%1000,rand()%1000);
    	      bool temp=boost::geometry::within(p_t, poly);
          }

          auto clock_end = std::chrono::high_resolution_clock::now();
          std::cout << "Time elapsed Boost is : " << std::chrono::duration_cast<std::chrono::nanoseconds>(clock_end - clock_start).count() << " microseconds.\n";        

          Eigen::MatrixXd point(100000,2);
          for(int kl=0;kl<100000;kl++)
          {
          	point(kl,0) = rand()%1000;
          	point(kl,1) = rand()%1000;
          }
          Eigen::MatrixXd stat(100000,1);
          // point<<test_.filtered[y].first,test_.filtered[y].second;
          clock_start = std::chrono::high_resolution_clock::now();
          InPoly(points,point,stat);
          clock_end = std::chrono::high_resolution_clock::now();
          std::cout << "Time elapsed In InPoly is : " << std::chrono::duration_cast<std::chrono::nanoseconds>(clock_end - clock_start).count() << " microseconds.\n";        
          return 0;

        }
        if(count<10||status) continue;
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
      imshow( "Before Filtering",mask ); 
      imshow( "Display window", new_mask );     
      imshow( "Real Image", test );   
      imshow("Dummy",dummy);
      // namedWindow( "Clusters", WINDOW_AUTOSIZE );
      // imshow( "Clusters", drawing );            
      std::cout<<count++<<endl;
      imshow( "Control", frame);
      char c=(char)waitKey(25);
      if(c==27)
          break;
  }
  cap.release();
  destroyAllWindows();
  return 0;
}