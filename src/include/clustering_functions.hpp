#ifndef CLUSTERING_FUNCTIONS_HPP
#define CLUSTERING_FUNCTIONS_HPP
#include <iostream>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <stack>
#include <unordered_set>
using namespace std;
using namespace cv;
using namespace Eigen;

class clustering
{
public:
  int M,N;
  MatrixXf an,ln;
  MatrixXi map;
  vector<pair<int,int>> m;
  VectorXi classified;
  MatrixXi ids;
  int max_class;
  clustering(MatrixXf& A,MatrixXf& B,MatrixXi& C,vector<pair<int,int>> &clus,MatrixXi id);
  vector<pair<int,int>> find_neighbors(int i,int j,double A, double l);
  VectorXi cluster();
  vector<unordered_set<int>> clusters;
  int find_unclassified(int start);
};

clustering::clustering(MatrixXf& A,MatrixXf& B,MatrixXi& C,vector<pair<int,int>>& clus,MatrixXi id)
{
  an=A;
  ln=B;
  M=C.rows();
  N=C.cols();
  map=C;
  m=clus;
  classified=VectorXi::Zero(clus.size());
  ids=id;
  max_class=0;
}

vector<pair<int,int>> clustering::find_neighbors(int a, int b, double A,double l)
{
  stack<pair<int,int>> checked;
  checked.push(make_pair(a,b));
  vector<pair<int,int>> t;
  int step=2;
  while(!checked.empty())
  {
    pair<int,int> point=checked.top();
    checked.pop();
    int x=point.first;
    int y=point.second;
    if(map(x,y)==0)
      continue;
    t.push_back(point);
    A=an(x,y);
    l=ln(x,y);
    map(x,y)=0;
    for(int i=-step;i<=step;i++)
    {
      for(int j=-step;j<=step;j++)
      {
        if(i==0||j==0||x+i>=M||y+j>=N||x+i<0||y+j<0||map(x+i,y+j)==0||abs(an(x+i,y+j)-A)*180/3.14>50||abs(l-ln(x+i,y+j))/(l+ln(x+i,y+j))>0.20)
          continue;
        checked.push(make_pair(x+i,y+j));
        // std::cout<<"Angle Difference: "<<an(x,y)-an(x+i,y+j)<<endl;
      }
    }
  }
  return t;
}

int clustering::find_unclassified(int start)
{
  for(int i=start;i<classified.size();i++)
  {
    if(classified(i)==0)
      return i;
  }
  return -1;
}

VectorXi clustering::cluster()
{
  VectorXi classes=VectorXi::Zero(classified.size());
  int class_no=1;
  int start=0;
  while(1)
  {
    int it=find_unclassified(start);
    if(it==-1)
      break;
    int x=m[it].first,y=m[it].second;
    vector<pair<int,int>> t=find_neighbors(x,y,an(x,y),ln(x,y));
    int points=0;
    for(auto x:t)
    {
        int id=ids(x.first,x.second);
        classified(id)=1;
    }
    if(t.size()>50)
    {
      unordered_set<int> set;
      for(auto x:t)
      {
          int id=ids(x.first,x.second);
          set.insert(id);
          classes(ids(x.first,x.second))=class_no;
      }
      clusters.push_back(set);
      class_no++;
    }
    else if(t.size()>20)
    {
      vector<Point2f> v;
      for(auto x:t)
      {
        v.push_back(Point2f(double(x.first),double(x.second)));
      }
      RotatedRect r=minAreaRect(v);
      double aspect_ratio=max(r.size.height,r.size.width)/min(r.size.height,r.size.width);
      if(aspect_ratio>3){
        unordered_set<int> set;
           for(auto x:t)
            {
              int id=ids(x.first,x.second);
              set.insert(id);
              classes(ids(x.first,x.second))=class_no;
            }
        clusters.push_back(set);
        class_no++;
      }
    }
  }
  max_class=class_no;
  return classes;
}

#endif
