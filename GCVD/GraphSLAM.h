// ************************ This is the GRAPHSLAM approach to Weighted least squares (WLS) **************************
// *
// *						CODE LOOSELY BASED ON E. ROSTEN'S TOON:WLS class 
// *
// *						Adapted by George Terzakis 2016
// *						
// *				WLS constructs and solves a Weighted LS non-liear system using the Fihgsre parametrization 
// *                                                       (see GraphSLAM by Thrun)

#ifndef GRAPHSLAM_H
#define GRAPHSLAM_H

//#include <TooN/TooN.h>
//#include <TooN/Cholesky.h>
//#include <TooN/helpers.h>

#include "Operators.h"

#include <cv.hpp>
#include <core.hpp>
#include <highgui.h>

#include <cmath>

using namespace cv;

namespace Optimization {

/// Performs Gauss-Newton weighted least squares computation.
/// @param Precision The numerical precision used (double, float etc)
template <typename Precision = float>
class WLS {

private:
	//Matrix<Size,Size,Precision> my_C_inv;
	cv::Mat_<Precision> Omega_; // probably the information matrix
	//Vector<Size,Precision> my_vector;
	cv::Mat_<Precision> ksi_; // This is probably the information vector. I am revaptising it to "ksi"
	//Decomposition<Size,Precision> my_decomposition;
	//Vector<Size,Precision> my_mu;
	cv::Mat_<Precision> mu_;
	
	cv::DecompTypes DecompositionType; // will be just passed to the inversion method
	int Dim;

public:

	/// Default constructor or construct with the number of dimensions for the Dynamic case
	WLS(int dim = 1, cv::DecompTypes decomposition = cv::DECOMP_CHOLESKY ) : Omega_( cv::Mat_<Precision>(dim, dim) ),
										 ksi_( cv::Mat_<Precision>(dim, 1) ),
										 mu_( cv::Mat_<Precision>(dim, 1) ),
										 Dim(dim)
	{
		clear();
	}

	/// Clear all the measurements and apply a constant regularisation term. 
	void clear() {
	  
		Omega_ = cv::Mat_<Precision>::zeros(Dim, Dim);
		ksi_ = cv::Mat_<Precision>::zeros(Dim, 1);
	}

	/// Applies a constant regularisation term. 
	/// Equates to a prior that says all the parameters are zero with \f$\sigma^2 = \frac{1}{\text{val}}\f$.
	/// @param val The information of the prior
	void add_prior(Precision val) {
		
	  for(int i=0; i<Omega_.rows; i++) 
	    Omega_(i,i)+=val;
		
	}
  
	/// Applies a regularisation term with a different strength for each parameter value. 
	/// Equates to a prior that says all the parameters are zero with \f$\sigma_i^2 = \frac{1}{\text{v}_i}\f$.
	/// @param v The vector of priors
	void add_prior(const cv::Mat_<Precision> &v) { // here v is a Dim x 1 matrix
		//SizeMismatch<Size,Size>::test(my_C_inv.num_rows(), v.size());
		for(int i=0; i<Omega_.rows; i++) 
			Omega_(i,i) += v(i, 0);
		
	}

	/// Applies a whole-matrix regularisation term. 
	/// This is the same as adding the \f$m\f$ to the inverse covariance matrix.
	/// @param m The inverse covariance matrix to add
	void add_prior(const cv::Mat_<Precision> &m) {
	  
		Omega_ += m;
	}

	/// Add a single measurement 
	/// @param m The value of the measurement (just one measurement)
	/// @param J The Jacobian for the measurement as a 1xDim matrix \f$\frac{\partial\text{m}}{\partial\text{param}_i}\f$
	/// @param weight The inverse variance of the measurement (default = 1)
	inline void add_mJ(Precision m, const cv::Mat_<Precision> &J, Precision weight = 1) {
		
		//Upper right triangle only, for speed
		for(int r=0; r < Omega_.rows; r++)
		{
			Precision Jw = weight * J(0, r);
			ksi_(r, 0) += m * Jw;
			for(int c = r; c < Omega_.rows; c++)
				Omega_(r, c) += Jw * J(0, c);
		}
	}

	/// Add multiple measurements at once (much more efficiently)
	/// @param m The measurements to add (Nx1 matrix)
	/// @param J The Jacobian matrix \f$\frac{\partial\text{m}_i}{\partial\text{param}_j}\f$ ( N x Dium)
	/// @param Qinv The inverse covariance of the measurement values
	inline void add_mJ(const cv::Mat_<Precision> &m,  // vector of measurements as an Nx1 matrix
			   const cv::Mat_<Precision> &J, // Jacobian (N x Dim) 
			   const cv::Mat_<Precision> &Qinv) {
		
		//const Matrix<Size,N,Precision> temp =  J * invcov;
		const cv::Mat_<Precision> temp =  J.t() * Qinv;  // this is a (Dim x N)*(N x N) = Dim x N matrix
		Omega_ += temp * J;
		ksi_ += temp * m;
	}

	/// Add multiple measurements at once (much more efficiently).
	/// @param m The measurements to add (Nx1 matrix)
	/// @param J The Jacobian matrix \f$\frac{\partial\text{m}_i}{\partial\text{param}_j}\f$ (N x Dim)
	/// @param Qinv The inverse covariance of the measurement values (NxN)
	inline void add_mJ_rows(const cv::Mat_<Precision> &m,  // vector of measurememts as Nx1 matrix
				const cv::Mat_<Precision> &J,
			        const cv::Mat_<Precision> &Qinv) {
	  
		const cv::Mat_<Precision> temp =  J.t() * Qinv;
		Omega_ += temp * J;
		ksi_ += temp * m;
	}

	/// Add a single measurement at once with a sparse Jacobian (much, much more efficiently)
	/// @param m The measurements to add
	/// @param J1 The first block of the Jacobian matrix \f$\frac{\partial\text{m}_i}{\partial\text{param}_j}\f$ Nx1 Jacobian vector as a cv::Mat_
	/// @param index1 starting index for the first block
	/// @param invcov The inverse covariance of the measurement values
	inline void add_sparse_mJ(const Precision m,
				  const cv::Mat_<Precision> &J1, // 1xN Jacobian vector as a cv::Mat_
				  const int index1,
				  const Precision weight = 1) {
		//Upper right triangle only, for speed
		for(int r=0; r < J1.cols; r++)
		{
			Precision Jw = weight * J1(0, r);
			ksi_[r+index1] += m * Jw;
			for(int c = r; c < J1.cols; c++)
				Omega_(r+index1, c+index1) += Jw * J1(0, c);
		}
	}

	/// Add multiple measurements at once with a sparse Jacobian (much, much more efficiently)
	/// @param m The measurements to add (Nx1 matrix)
	/// @param J1 The first block of the Jacobian matrix \f$\frac{\partial\text{m}_i}{\partial\text{param}_j}\f$
	/// (N x S1) jacobian of S1 variables over N measurements
	/// @param index1 starting index for the first block
	/// @param Qinv The inverse covariance of the measurement values
	inline void add_sparse_mJ_rows(const cv::Mat_<Precision> &m,
				       const cv::Mat_<Precision> &J1, 
				       const int index1,
				       const cv::Mat_<Precision> &Qinv) {
		const cv::Mat_<Precision> temp1 = J1.t() * Qinv; // this is a S1xN matrix now
		const int size1 = J1.cols;
		//my_C_inv.slice(index1, index1, size1, size1) += temp1 * J1;
		cv::Mat_<Precision> OmegaBlockHead = Omega_(cv::Range(index1, index1 + size1), cv::Range(index1, index1 + size1) ); 
		cv::Mat_<Precision> updatedBlock = OmegaBlockHead + temp1 * J1;
		updatedBlock.copyTo( OmegaBlockHead );
		
		//my_vector.slice(index1, size1) += temp1 * m;
		cv::Mat_<Precision> ksiBlockHead = ksi_(cv::Range(index1 , index1 + size1), cv::Range(0, 1) );
		cv::Mat_<Precision> updatedKsiBlock = ksiBlockHead + temp1 * m;
		updatedKsiBlock.copyTo( ksiBlockHead );
	  }

	/// Add multiple measurements at once with a sparse Jacobian (much, much more efficiently)
	/// @param m The measurements to add (Nx1 Mat_)
	/// @param J1 The first block of the Jacobian matrix \f$\frac{\partial\text{m}_i}{\partial\text{param}_j}\f$
	/// Jacobian is Nxnumber of Variables == Sz1 
	/// @param index1 starting index for the first block
	/// @param J2 The second block of the Jacobian matrix \f$\frac{\partial\text{m}_i}{\partial\text{param}_j}\f$
	/// N x Sz2
	/// @param index2 starting index for the second block
	/// @param Qinv The inverse covariance of the measurement values
	inline void add_sparse_mJ_rows(const cv::Mat_<Precision> &m,
				       const cv::Mat_<Precision> &J1, 
				       const int index1,
				       const cv::Mat_<Precision> &J2, 
				       const int index2,
				       const cv::Mat_<Precision> &Qinv) {
		
		const cv::Mat_<Precision> temp1 = J1.t() * Qinv;
		const cv::Mat_<Precision> temp2 = J2.t() * Qinv;
		const cv::Mat_<Precision> mixed = temp1 * J2;
		const int size1 = J1.cols;
		const int size2 = J2.cols;
		//my_C_inv.slice(index1, index1, size1, size1) += temp1 * J1;
		cv::Mat_<Precision> OmegaBlock1Head = Omega_(cv::Range(index1, index1 + size1), cv::Range(index1, index1 + size1));
		cv::Mat_<Precision> updatedBlock1 = OmegaBlock1Head + temp1 * J1;
		updatedBlock1.copyTo( OmegaBlock1Head);
		
		//my_C_inv.slice(index2, index2, size2, size2) += temp2 * J2;
		cv::Mat_<Precision> OmegaBlock2Head = Omega_( cv::Range(index2, index2 + size2), cv::Range(index2, index2 + size2));
		cv::Mat_<Precision> updatedBlock2 = OmegaBlock2Head + temp2 * J2;
		updatedBlock2.copyTo( OmegaBlock2Head);
		
		//my_C_inv.slice(index1, index2, size1, size2) += mixed;
		cv::Mat_<Precision> OmegaBlock12Head = Omega_( cv::Range(index1, index1 + size1), cv::Range(index2, index2 + size2));
		cv::Mat_<Precision> updatedBlock12 = OmegaBlock12Head + mixed;
		updatedBlock12.copyTo( OmegaBlock12Head);
		
		//my_C_inv.slice(index2, index1, size2, size1) += mixed.T();
		cv::Mat_<Precision> OmegaBlock21Head = Omega_( cv::Range(index2, index2 + size2), cv::Range(index1, index1 + size1));
		cv::Mat_<Precision> updatedBlock21 = OmegaBlock21Head + mixed.t();
		updatedBlock21.copyTo( OmegaBlock21Head);
		
		//my_vector.slice(index1, size1) += temp1 * m;
		cv::Mat_<Precision> ksiBlock1Head = ksi_( cv::Range(index1 , index1 + size1), cv::Range(0, 1) );
		cv::Mat_<Precision> updatedKsiBlock1 = ksiBlock1Head + temp1 * m;
		updatedKsiBlock1.copyTo( ksiBlock1Head );
		
		//my_vector.slice(index2, size2) += temp2 * m;
		cv::Mat_<Precision> ksiBlock2Head = ksi_( cv::Range(index2 , index2 + size2), cv::Range(0, 1) );
		cv::Mat_<Precision> updatedKsiBlock2 = ksiBlock2Head + temp2 * m;
		updatedKsiBlock2.copyTo( ksiBlock2Head );
		
	}

	/// Process all the measurements and compute the weighted least squares set of parameter values
	/// stores the result internally which can then be accessed by calling get_mu()
	void compute() {
	
		//Copy the upper right triangle to the empty lower-left.
		for(int r=1; r < Omega_.rows; r++)
			for(int c=0; c < r; c++)
				Omega_(r, c) = Omega_(c, r);

		//my_decomposition.compute(my_C_inv);
		//my_mu=my_decomposition.backsub(my_vector);
		mu_ = Omega_.inv(DecompositionType) * ksi_;
	}

	/// Combine measurements from two WLS systems
	/// @param meas The measurements to combine with
	void operator += (const WLS& meas) {
		
	    ksi_ += meas.ksi_;
	    Omega_ += meas.Omega_;
	}

	/// Returns the inverse covariance matrix
	cv::Mat_<Precision>& get_Omega() {return Omega_;}
	/// Returns the inverse covariance matrix
	const cv::Mat_<Precision>& get_Omega() const {return Omega_;}
	cv::Mat_<Precision>& get_mu(){return mu_;}  ///<Returns the update. With no prior, this is the result of \f$J^\dagger e\f$.
	const cv::Mat_<Precision>& get_mu() const {return mu_;} ///<Returns the update. With no prior, this is the result of \f$J^\dagger e\f$.
	cv::Mat_<Precision>& get_ksi(){return ksi_;} ///<Returns the  vector \f$J^{\mathsf T} e\f$
	const cv::Mat_<Size,Precision>& get_ksi() const {return ksi_;} ///<Returns the  vector \f$J^{\mathsf T} e\f$
	cv::DecompTypes& get_decomposition(){return DecompositionType;} ///< Return the decomposition object used to compute \f$(J^{\mathsf T}  J + P)^{-1}\f$
	const cv::DecompTypes& get_decomposition() const {return DecompositionType;} ///< Return the decomposition object used to compute \f$(J^{\mathsf T}  J + P)^{-1}\f$


};

}

#endif