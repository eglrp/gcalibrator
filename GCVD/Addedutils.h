// some simple added utils that existed in CVD 
// and have to be added here...

#ifndef ADDED_UTILS_H
#define ADDED_UTILS_H

#include <cv.hpp>
#include <core.hpp>
#include <cxcore.hpp>
#include <highgui.hpp>
#include "Operators.h"

//#include "scalar_convert.h"

namespace CvUtils {
  
static void pause(cv::Mat img)
{
  cv::namedWindow("pause");
  cv::imshow("pause", img);
  cv::waitKey(-1);
}

static void pause()
{
cv::waitKey(-1);
}

template <typename T, typename S, typename Precision> 
inline void sample(const cv::Mat_<S> &im, 
		    Precision x, // x - coordinate to sample in the "in" matrix
		    Precision y, // y - coordinate to sample in the "in" matrix
		    T* result    // location to bestow the (potentially) multichanneled samples
		  );
  
inline bool in_image_with_border(cv::Point2i pos, cv::Mat im, cv::Size2i border)
{
  return ( pos.x - border.width >= 0 ) && 
	 ( pos.y - border.height >= 0 ) &&
	 ( pos.x + border.width  < im.cols ) &&
         ( pos.y + border.height < im.rows );
};

inline bool in_image_with_border(int row, int col, cv::Mat im, int bwidth, int bheight)
{
  return ( col - bwidth >= 0 ) && 
	 ( row - bheight >= 0 ) &&
	 ( col + bwidth  < im.cols ) &&
         ( row + bheight < im.rows );
};

// 2x2 inverse!!!!!
template<typename T>
inline cv::Mat_<T> M2Inverse(const cv::Mat_<T> &m)
{
  cv::Mat_<T> m2Res(2,2);
  double dDet = m[0][0] * m[1][1] - m[1][0] * m[0][1];
  assert(dDet!=0.0);
  double dInverseDet = 1.0 / dDet;
  m2Res(0, 0) = m(1, 1) * dInverseDet;
  m2Res(1, 1) = m(0, 0) * dInverseDet;
  m2Res(1, 0) = -m(1, 0) * dInverseDet;
  m2Res(0, 1) = -m(0, 1) * dInverseDet;
  
  return m2Res;
}


// This is the "templated" version of Rosten's code using OpenCV.
// I am using his transform function instead of OpenCV's WarpAffine
// because the devil hides in the details!
 template <typename Tin, typename Tout, typename P>
int transform(const cv::Mat_<Tin> &in, 
	      cv::Mat_<Tout> &out, 
	      const cv::Mat_<P> &M,   // a 2x2 matrix (affine scaling) 
	      const cv::Vec<P, 2> &inOrig, // a 2x1 vector (translation)
	      const cv::Vec<P, 2> &outOrig, // and another 2x1 vector!
	      const Tout defaultValue = Tout())   // default value for boundary pixels
{
    const int w = out.cols, h = out.rows, iw = in.cols, ih = in.rows; 
    const cv::Vec<P, 2> across( M(0, 0), M(1, 0) ); // column #1 of M
    const cv::Vec<P, 2> down( M(0, 1), M(1, 1) );  // column #2 of M
    const cv::Vec<P, 2> p0 = inOrig - cv::Vec<P, 2>( M(0,0) * outOrig[0] + M(0,1) * outOrig[1], 
						     M(1,0) * outOrig[0] + M(1,1) * outOrig[1] );
    // trying to avoid the -UNTESTED- operator overloads for scalr * vector, just in case (they do work though...).
    const cv::Vec<P, 2> p1( p0[0]+ w*across[0], 
			    p0[1]+ w*across[1]
			  );
    const cv::Vec<P, 2> p2( p0[0] + h*down[0],
			    p0[1] + h*down[1] 
			  );
    const cv::Vec<P, 2> p3( p0[0] + w*across[0] + h*down[0], 
			    p0[1] + w*across[1] + h*down[1]
			  );
        
    // ul --> p0
    // ur --> w*across + p0
    // ll --> h*down + p0
    // lr --> w*across + h*down + p0
    P min_x = p0[0], min_y = p0[1];
    P max_x = min_x, max_y = min_y;
   
    // Minimal comparisons needed to determine bounds
    if (across[0] < 0)
	min_x += w*across[0];
    else
	max_x += w*across[0];
    if (down[0] < 0)
	min_x += h*down[0];
    else
	max_x += h*down[0];
    if (across[1] < 0)
	min_y += w*across[1];
    else
	max_y += w*across[1];
    if (down[1] < 0)
	min_y += h*down[1];
    else
	max_y += h*down[1];
   
    // This gets from the end of one row to the beginning of the next
    const cv::Vec<P, 2> carriage_return( down[0] - w*across[0],
					 down[1] - w*across[1] 
				       );

    //If the patch being extracted is completely in the image then no 
    //check is needed with each point.
    if (min_x >= 0 && min_y >= 0 && max_x < iw-1 && max_y < ih-1) 
    {
	cv::Vec<P, 2> p = p0;
	for (int r=0; r<h; ++r, p+=carriage_return)
	    for (int c=0; c<w; ++c, p+=across) 
		sample(in,p[0],p[1], &out(r, c) );
	return 0;
    } 
    else // Check each source location
    {
	// Store as doubles to avoid conversion cost for comparison
	const P x_bound = iw-1;
	const P y_bound = ih-1;
	int count = 0;
	cv::Vec<P, 2> p = p0;
	for (int r=0; r<h; ++r, p+=carriage_return) {
	    for (int c=0; c<w; ++c, p+=across) {
		//Make sure that we are extracting pixels in the image
		if (0 <= p[0] && 0 <= p[1] &&  p[0] < x_bound && p[1] < y_bound)
		    sample(in,p[0],p[1], (Tout*)out.ptr(r, c) );
		else {
            out[r][c] = defaultValue;
		    ++count;
		}
	    }
	}
	return count;
    }
}




template <typename T, typename S, typename Precision> 
inline void sample(const cv::Mat_<S> &im, 
		    Precision x, // x - coordinate to sample in the "in" matrix
		    Precision y, // y - coordinate to sample in the "in" matrix
		    T* result    // location to bestow the (potentially) multichanneled samples
		  )
{
  int numChannels = im.channels();
  
  const int lx = (int)x;
  const int ly = (int)y;
  x -= lx;
  y -= ly;
  // Summing per channel. I think that this template concoction should work in any case...
  T* rptr = result;
  S* imptr = (S*)im.ptr(ly , lx);
  S* imptr01 = (S*)im.ptr(ly , lx+1);
  S* imptr10 = (S*)im.ptr(ly+1 , lx);
  S* imptr11 = (S*)im.ptr(ly + 1, lx+ 1);
  
  for(unsigned int i = 0; i < numChannels; i++)
    rptr[i] =  (1-y)*( (1-x)*imptr[i] + x*imptr01[i] ) + y * ( (1-x)*imptr10[i] + x*imptr11[i] ) ; 
 
}


// Determinant of 2x2
template<typename T>
inline T M2Det(cv::Mat_<T> m)
{
  return m(0, 0) * m(1, 1)  - m(0, 1) * m(1, 0);
}

// Determinant of 3x3
template<typename T>
inline T M3Det(cv::Mat_<T> m )
{
  return  
    m(0, 0) * ( m(1, 1) * m(2, 2)  - m(1, 2) * m(2, 1) ) - 
    m(0, 1) * ( m(1, 0) * m(2, 2)  - m(1, 2) * m(2, 0) ) + 
    m(0, 2) * ( m(1, 0) * m(2, 1)  - m(1, 1) * m(2, 0) );
}


 
template<typename T>
inline T M3Det(cv::Mat_<T> m );




template <typename P, int Sz>
inline cv::Vec<P, Sz> normalize(const cv::Vec<P, Sz> &v) {
 P n = cv::norm(v);
 
  return n > 0 ? v / n : v; 
}

template <typename P>
inline cv::Mat_<P> normalize(const cv::Mat_<P> &m) {
  P n = cv::norm(m);
  
  return n > 0 ? m / n : m; 
}


// wouldn't use this unless I really needed it...
template <typename P, int Sz1, int Sz2>
const cv::Vec<P, Sz2>& slice(const cv::Vec<P,Sz1> &v, int offset) {
  cv::Vec<P, Sz2> ret;
  for (int i = 0; i < ret.rows; i++)
    ret[i] = v[i+offset];
  
  return ret;
}


// conversion from cv::mat_ to cv::vector
template<typename P, int Sz> 
inline cv::Vec<P, Sz>  mat2Vec(const cv::Mat_<P> &m) {
  cv::Vec<P, Sz> ret;
  
  int sz = m.rows >= m.cols ? m.rows : m.cols; 
  bool columnVec = m.rows == sz;
  for (int i = 0; i<sz; i++)
    ret[i] = columnVec ? m(i, 0) : m(0, i);
  
  return ret;
}


// conversion from cv::Vec to cv::Mat
template<typename P, int Sz> 
const cv::Mat_<P> mat2Vec(const cv::Vec<P, Sz> &v) {
  cv::Mat_<P> ret(v.rows, 1);
  
  for (int i = 0; i<v.rows; i++)
    ret[i] = v[i];
  
  return ret;
}


};




#endif