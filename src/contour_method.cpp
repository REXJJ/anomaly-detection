#include "cv.h"
#include "highgui.h"
#include "include/reference_bs.hpp"

using namespace cv;
using namespace std;

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

      image_params test_(test,2,7);
      test_.stat_calculation();
      Mat mask=get_mask(image.mean_m,test_.mean_m,image.std_m,test_.std_m,image.mean_A,test_.mean_A,20.0);
      namedWindow( "Display window", WINDOW_AUTOSIZE );
      imshow( "Display window", mask );     
      namedWindow( "Real Image", WINDOW_AUTOSIZE );
      imshow( "Real Image", test );             
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