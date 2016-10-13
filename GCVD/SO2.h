#ifndef SO2_H
#define SO2_H

//#include <TooN/TooN.h>
//#include <TooN/helpers.h>
#include "Operators.h"
#include "Addedutils.h"

#include "cv.hpp"
#include "core.hpp"
#include "highgui.hpp"

#include <math.h>


namespace RigidTransforms {

template <typename Precision> class SO2;
template <typename Precision> class SE2;
//template <typename Precision> class SIM2;

template<typename Precision> inline std::istream & operator >> (std::istream&, SO2<Precision>& );
template<typename Precision> inline std::istream & operator >> (std::istream&, SE2<Precision>& );
//template<typename Precision> inline std::istream & operator >> (std::istream&, SIM2<Precision>& );

/// Class to represent a two-dimensional rotation matrix. Two-dimensional rotation
/// matrices are members of the Special Orthogonal Lie group SO2. This group can be parameterised with
/// one number (the rotation angle).
template<typename Precision = float>
class SO2 {
	friend std::istream& operator>> <Precision>(std::istream&, SO2& );
	friend std::istream& operator>> <Precision>(std::istream&, SE2<Precision>& );
	//friend std::istream& operator>> <Precision>(std::istream&, SIM2<Precision>& );

private: 
	struct Invert {};
	inline SO2(const SO2 &so2, const Invert&) : mat_(so2.mat_.t()) {}
	template <typename PA, typename PB>
	inline SO2(const SO2<PA>& a, const SO2<PB>& b) : mat_(a.get_matrix()*b.get_matrix()) {}

	cv::Mat_<Precision> mat_; // the 2x2 matrix containing the transformation.
	
	
public:
	
	
	/// Default constructor. Initialises the matrix to the identity (no rotation)
	SO2() :mat_(cv::Mat_<Precision>::eye(2, 2))
	{}
	
	
	/// Construct from a rotation matrix.
	SO2(const cv::Mat_<Precision> &rhs) {  
		*this = rhs; 
		coerce(); // skip for now...
	}

	// Construct from an angle (Lie logarithm).
	explicit SO2(const Precision angle) { *this = exp(angle); }
  
	/// Assigment operator from a general matrix. This also calls coerce()
	/// to make sure that the matrix is a valid rotation matrix.
	template <typename P> 
	SO2& operator =(const cv::Mat_<P> &R){
		mat_(0,0) = (Precision)R(0,0);
		mat_(0,1) = (Precision)R(0,1);
		mat_(1,0) = (Precision)R(1,0);
		mat_(1,1) = (Precision)R(1,1);
		
		coerce(); // skip for now... 
		return *this;
	}
	
	
	// some helper functions...
	cv::Vec<Precision, 2> colAt(int index) {
	  return cv::Vec<Precision, 2>(mat_(0, index), 
				       mat_(1, index) );
	}
	
	cv::Vec<Precision, 2> rowAt(int index) {
	  return cv::Vec<Precision, 2>(mat_(index, 0), 
				       mat_(index, 1) );
	}
	
	void assignRowAt(int index, const cv::Vec<Precision, 2> &v) {
	  mat_(index, 0) = v[0];
	  mat_(index, 1) = v[1];
	  
	}
	
	void assignColAt(int index, const cv::Vec<Precision, 2> &v) {
	  mat_(0, index) = v[0];
	  mat_(1, index) = v[1];
	  
	}
	
	
	
	/// Modifies the matrix to make sure it is a valid rotation matrix. Gram-Schmidt orthogonalization
	void coerce(){
		// 1. normalize first row
		assignRowAt(0, normalize(rowAt(0)));
		
		// 2. Project 2nd row on the 1st and take the normalized difference of the 2nd from the prjection
		//mat_(1, 0) -= mat_[0] * (mat_[0]*mat_[1]);
		assignRowAt(1, normalize( rowAt(1) - (rowAt(1)*rowAt(0))*rowAt(0) ) );
		
	}

	/// Exponentiate an angle in the Lie algebra to generate a new SO2.
	inline static SO2 exp(const Precision &d) {
		SO2<Precision> result;
		
		result.mat_(0, 0) = result.mat_(1, 1) = cos(d);
		result.mat_(1, 0) = sin(d);
		result.mat_(0, 1) = -result.mat_(1, 0);
		
		return result;
	}

	/// extracts the rotation angle from the SO2
	Precision ln() const { return atan2(mat_(1, 0), mat_(0, 0)); }

	/// Returns the inverse of this matrix (=the transpose, so this is a fast operation)
	SO2 inverse() const { return SO2(*this, Invert()); }

	

	
	/// Self right-multiply by another rotation SO2
	template <typename P>
	SO2& operator *=(const SO2<P> &right){
		mat_ = mat_ * right.get_matrix();
		
		return *this;
	}

	/// Right-multiply by another SO2
	template <typename P>
	SO2<typename MyOperatorOverloads::MultiplyType<Precision, P>::type>& operator *(const SO2<P> &right) const { 
		 
	    typedef typename MyOperatorOverloads::MultiplyType<Precision, P>::type P0;
	    
	    return SO2<P0>( this->mat_ * right->get_matrix() ); 
	}

	/// Returns the SO2 as a Matrix<2>
	const cv::Mat_<Precision>& get_matrix() const {return mat_;}
	
	cv::Mat_<Precision>& get_matrix() {return mat_;} // IMPORTANT OVERLOAD!!!!!!!!

	/// returns Lie generator matrix (skew symmetric matrix)
	static cv::Mat_<Precision> generator() {
		
		cv::Mat_<Precision> result(2,2);
		result(0, 0) = 0; result(0, 1) = -1;
		result(1, 0) = 1; result(1, 1) = 0;
		
		return result;
	}


}; // *************** Here ends the class SO2 - Here ends the class SO2, in order to define binary operator overloads ***************

  
} // Ends RigidTransfomrs - Ends RigidTransfomrs - Ends RigidTransfomrs - Ends RigidTransfomrs - Ends RigidTransfomrs **********

/// Write an SO2 to a stream 
template <typename Precision>
inline std::ostream& operator << (std::ostream &os, const RigidTransforms::SO2<Precision> &right) {
  
	return os << right.get_matrix();
}

/// Read from SO2 to a stream 
template <typename Precision>
inline std::istream& operator>>(std::istream &is, RigidTransforms::SO2<Precision> &right) {
	is >> right.mat_;
	right.coerce(); // skip for now
	
	return is;
}


/// Right-multiply by a 2-Vector
template<typename P, typename PV>
inline cv::Vec<typename MyOperatorOverloads::MultiplyType<P, PV>::type, 2> operator *(const RigidTransforms::SO2<P> &so2, const cv::Vec<PV, 2> &v) {
  
  typedef typename MyOperatorOverloads::MultiplyType<P, PV>::type P0;
  
  cv::Mat_<P0> temp = so2.get_matrix() * v; // a 2x1 matrix
  
  return cv::Vec<P0, 2>( temp(0, 0), temp(1, 0) );
}

/// Left-multiply by a Vector // this basically results in a vector u = R^T * v
template<typename PV, typename P>
inline cv::Vec<typename MyOperatorOverloads::MultiplyType<PV,P>::type, 2> operator *(const cv::Vec<PV,2> &v, const RigidTransforms::SO2<P> &so2) {

  
  typedef typename MyOperatorOverloads::MultiplyType<P, PV>::type P0;
  
  cv::Mat_<P0> temp = v * so2.get_matrix(); // a 2x1 matrix if all went well...
  
  return cv::Vec<P0, 2>( temp(0,0), temp(1, 0) );
}

/// Right-multiply by a Matrix
template <typename P, typename PM> 
inline cv::Mat_<typename MyOperatorOverloads::MultiplyType<P,PM>::type> operator *(const RigidTransforms::SO2<P> &so2, const cv::Mat_<PM> &M){
	
  return so2.get_matrix() * M;
}

/// Left-multiply by a Matrix
template <typename PM, typename P>
inline cv::Mat_<typename MyOperatorOverloads::MultiplyType<PM,P>::type> operator *(const cv::Mat_<PM> M, const RigidTransforms::SO2<P> &so2) {
	
  return M * so2.get_matrix();
}



#endif