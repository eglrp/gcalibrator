#ifndef CVD_IMAGE_INTERPOLATE_H
#define CVD_IMAGE_INTERPOLATE_H

//#include <TooN/TooN.h>

//#include <cvd/image.h>
//#include <cvd/internal/pixel_operations.h>
//#include <cvd/internal/rgb_components.h>
//#include <cvd/vector_image_ref.h>

#include <cv.hpp>
#include <highgui.hpp>
#include <core.hpp>

#include "Addedutils.h"

#include <math.h>

namespace CvUtils
{
	///Classes used to specify the interpolation type
	///for image_interpolate.

	namespace Interpolate
	{
	         /** This does not interpolate: it uses the nearest neighbour.
		     
		 */
		class NearestNeighbour{};

		/** This class is for bilinear interpolation.
		    
		    Define \f$p' = ( \operatorname{floor}\ x, \operatorname{floor}\ y)\f$ and
		    \f$\delta = p - p'\f$
		    
		    4 pixels in a square with \f$p'\f$ in the top left corner are taken:
		    \f[\begin{array}{rl}
		    a =& I(p')\\   
		    b =& I(p' + (1,0))\\  
		    c =& I(p' + (0,1))\\  
		    d =& I(p' + (1,1))
		    \end{array}
		    \f]
		    
		    The interpolated value, \f$v\f$, is 
		    \f[v = (1-\delta_y)((1-\delta_x)a + \delta_xb) + \delta_y((1-\delta_x)c + \delta_xd)\f]
		*/
		class Bilinear{};

		/**
 		  This class is for bicubic (not bicubic spline) interpolation.
		  \f[ v = \sum_{m=-1}^2\sum_{n=-1}^2 I(x' + m, y' + n)r(m - \delta_x)r(\delta_y-n) \f]
  		  where:
		    \f[\begin{array}{rl}
		    r(x) =& \frac{1}{6}\left[ p(x+2)^3 - 4p(x+1)^3 + 6p(x)^3 - 4p(x-1)^3 \right]\\
		    p(x) =& \begin{cases}x&x>0\\0&x \le 0\end{cases}
		    \end{array}\f]                                                             
		
		    This algorithm is described in http://astronomy.swin.edu.au/~pbourke/colour/bicubic/ 
		*/
		class Bicubic{};
	};


	// decided that interpolation values are always single precision and that's it.
	template<class I, typename T> class image_interpolate;

	//Zero order (nearest neighbour)

	template<typename T>
	class image_interpolate<Interpolate::NearestNeighbour, T>
	{
		private:
			const cv::Mat_<T> *im;
			const unsigned int numChannels;

			int round(double d) const
			{
				if(d < 0)
					return (int)std::ceil(d - .5);
				else
					return (int)std::floor(d + .5);
			}

			template<typename P>
			cv::Point2i to_ir(const cv::Vec<P, 2> &v) const
			{
			  return cv::Point2i(round(v[0]), round(v[1]));
			}

			//typedef typename Pixel::traits<C>::float_type FT;
			//typedef class cv::Scalar_<C> FT;
	
		public:
			image_interpolate(const cv::Mat_<T> &i) : im(&i), numChannels(i.channels()) {}
			
			template<typename P>
			bool in_image(const cv::Vec<P, 2> &pos) const
			{
				return (pos[0] >= 0) && (pos[1] >= 0) &&
				       (pos[0] < im->cols ) && (pos[1] < im->rows);
			}

			template<typename P>
			cv::Vec4d operator[](const cv::Vec<P, 2> &pos) const
			{
				cv::Vec4d ret;
				
				for (int i = 0; i<numChannels; i++) {
				 T* pt = (T*)im->ptr( (int)std::round(pos[1]), (int)std::round(pos[0]) );
				  ret[i] = (double)pt[i]; 
				  
				}
				
				return ret;
			}

			template<typename P>
			cv::Vec<P, 2> min() const
			{
				return cv::Vec<P, 2>( 0, 0);
			}

			template<typename P>
			cv::Vec<P, 2> max() const
			{
				return cv::Vec<P, 2>(im->cols, im->rows);
			}

			
	};

	template<typename T> 
	class image_interpolate<Interpolate::Bilinear, T>
	{
		private:
			const cv::Mat_<T> *im;
			const unsigned int numChannels;

			template<typename P>
			cv::Vec<P, 2> floor(const cv::Vec<P, 2> &v) const
			{
				return cv::Vec<P, 2>( (P)std::floor(v[0]), (P)std::floor(v[1]));
			}

			template<typename P>
			cv::Vec<P, 2> ceil(const cv::Vec<P, 2> &v) const
			{
				return cv::Vec<P, 2>( (P)std::ceil(v[0]), (P)std::ceil(v[1]));
			}

			//typedef typename Pixel::traits<T>::float_type FT;
			//typedef class cv::Scalar_<T> FT;
		public:
			image_interpolate(const cv::Mat_<T> &i) :im(&i), numChannels(i.channels()) {}

			template<typename P>
			bool in_image(const cv::Vec<P, 2> &pos) const
			{
				//return im->in_image(ir(floor(pos))) && im->in_image(ir(ceil(pos)));
				  return (std::floor(pos[0]) >= 0) && (std::floor(pos[1]) >= 0) &&
					 (std::floor(pos[0]) < im->cols) && (std::floor(pos[1]) < im->rows) &&
					 (std::ceil(pos[0]) >= 0) && (std::ceil(pos[1]) >= 0) &&
					 (std::ceil(pos[0]) < im->cols) && (std::ceil(pos[1]) < im->rows) ;
					 
			  
			}

			
			template<typename P>
			cv::Vec4d operator[](const cv::Vec<P, 2> &pos) const
			{
				cv::Vec<P, 2> delta =  pos - floor(pos);

				
				cv::Point2i p = cv::Point2i((int)std::floor(pos[0]), (int)std::floor(pos[1]));
				
				P x = delta[0];
				P y = delta[1];

				
				// interpolation ALWAYS returns double precision
				cv::Vec4d ret(0, 0, 0, 0);
				
				
				//for(unsigned int i=0; i < Pixel::Component<T>::count; i++)
				T* ptr = (T*)im->ptr(p.y, p.x);
				T* ptr01 = (T*)im->ptr(p.y, p.x+1);
				T* ptr10 = (T*)im->ptr(p.y + 1, p.x);
				T* ptr11 = (T*)im->ptr(p.y + 1, p.x + 1);
				
				for(unsigned int i=0; i < numChannels; i++)
				{
					double a, b=0, c=0, d=0;
					
					
					//a = im->at<cv::Scalar_<4>>(p.y, p.x)[i] * (1-x) * (1-y);
					a = (double)( ptr[i] ) * (double)( (1-x) * (1-y) );
					if(x != 0) //b = im->at<cv::Scalar>(p.y, p.x+1)[i] * x * (1-y);
					    b = (double)( ptr01[i] ) * (double)( x * (1-y) );
					if(y != 0)
					//c = im->at<cv::Scalar>(p.y+1, p.x)[i] * (1-x) * y;
					  c = (double)( ptr10[i] ) * (double)( (1-x) * y );
					if(x !=0 && y != 0)
						//d = im->at<cv::Scalar>(p.y+1, p.x+1)[i] * x * y;
					  d = (double)( ptr11[i] ) * (double) ( x * y );
					
					ret[i] = a + b + c + d;
					//ret.val[i] = a + b + c + d;
				}

				return ret;
			}

			template<typename P>
			cv::Vec<P, 2> min() const
			{
				return cv::Vec<P,2>( 0, 0);
			}

			template<typename P>
			cv::Vec<P, 2> max() const
			{
				return cv::Vec<P, 2>(im->cols, im->rows);
			}

	};


	template<typename T> class image_interpolate<Interpolate::Bicubic, T>
	{
		private:
			const cv::Mat_<T> *im;
			const unsigned int numChannels;
			
			template<typename P>
			P p(P f) const
			{
				return f <0 ? 0 : f;
			}
			
			template<typename P>
			P r(P x) const
			{
				return (  pow(p(x+2), 3) - 4 * pow(p(x+1),3) + 6 * pow(p(x), 3) - 4* pow(p(x-1),3))/6;
			}

			//typedef class cv::Scalar_<float> FT;

		public:
			image_interpolate(const cv::Mat_<T> &i) :im(&i), numChannels(i.channels()) {}

			template<typename P>
			bool in_image(const cv::Vec<P, 2> &pos) const
			{
				return pos[0] >= 1 && pos[1] >=1 && pos[0] < im->cols-2 && pos[1] < im->rows - 2;
			}

			template<typename P>
			cv::Vec4d operator[](const cv::Vec<P, 2> &pos) const
			{
				int x = (int)std::floor(pos[0]);
				int y = (int)std::floor(pos[1]);
				P dx = pos[0] - x;
				P dy = pos[1] - y;

				//Algorithm as described in http://astronomy.swin.edu.au/~pbourke/colour/bicubic/
				
				// Inetrpolation ALWAYS returns doubles
				cv::Vec4d ret(0, 0, 0, 0);
				
				for(unsigned int i=0; i < numChannels; i++) {
				  
					  double s=0;
  
					  for(int m = -1; m < 3; m++)
					    for(int n = -1; n < 3; n++) {
					      T* ptr = (T*)im->ptr(y+n, x+m);
					      s += ( (double)ptr[i] ) * (double)( r(m - dx) * r(dy-n) ) ;
						
					    }
					  ret[i]= s;
					  
				  }
				 
				return ret;
			}

			
			template<typename P>
			cv::Vec<P, 2> min() const
			{
			  return cv::Vec<P, 2>( 1, 1);
			}

			template<typename P>
			cv::Vec<P, 2> max() const
			{
				return cv::Vec<P, 2>( im->cols - 2, im->rows - 2);
			}

	};
	

}

#endif