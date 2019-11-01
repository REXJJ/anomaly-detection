#include "cv.h"
#include "highgui.h"
#include "include/reference_bs.hpp"

using namespace cv;
using namespace std;

int mag_mean=0;
int mag_sd=0;
string window_detection_name;

static void mean_magnitude(int, void *)
{
    setTrackbarPos("magnitude_mean", window_detection_name, mag_mean);
}
static void sd_magnitude(int, void *)
{
    setTrackbarPos("magnitude_sd", window_detection_name, mag_sd);
}



int main( int argc, char** argv )
{

  window_detection_name="Control";
  namedWindow(window_detection_name,WINDOW_AUTOSIZE);

  createTrackbar("magnitude_mean", window_detection_name, &mag_mean,100,mean_magnitude);
  createTrackbar("magnitude_sd", window_detection_name, &mag_sd,100,sd_magnitude);

  VideoCapture cap("/home/rex/Desktop/Images/ICRA/test1.mp4"); 
    
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  Rect ROI(704,243,723,650); 
  int k=2,kernel_size=7; 
  Mat reference,test;
  reference=imread("/home/rex/Desktop/Images/ICRA/reference.jpg",0);
  reference=reference(ROI);
  image_params image(reference,k,kernel_size);
  image.stat_calculation();
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
      std::cout<<mag_sd<<" "<<mag_mean<<endl;
      image_params test_(test,k,kernel_size);
      test_.stat_calculation();
      vector<double> params={double(mag_mean),double(mag_sd),20,10};
      Mat mask=get_mask(image.mean_m,test_.mean_m,image.std_m,test_.std_m,image.mean_A,test_.mean_A,image.std_A,test_.std_A,params);
      imshow(window_detection_name, mask );     
      namedWindow( "Real Image", WINDOW_AUTOSIZE );
      imshow( "Real Image", test );             
      std::cout<<count++<<endl;
      char c=(char)waitKey(25);
      if(c==27)
          break;
  }
  cap.release();
  destroyAllWindows();
  return 0;
}