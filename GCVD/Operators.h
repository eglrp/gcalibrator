#ifndef OPERATORS_H
#define OPERATORS_H



#include "../OpenCV.h"

#include <limits.h>
using namespace std;
using namespace cv;



namespace MyOperatorOverloads {

  
//Automatic type deduction of return types
///This function offers to return a value of type C. This function
///is not implemented anywhere, the result is used for type deduction.
template<class C> C gettype();
	

// This is a BRILIANT way of working out and defining the type triplets of pairwise operations
template<class L, class R> struct AddType      { typedef decltype (gettype<L>()+gettype<R>()) type;};
template<class L, class R> struct SubtractType { typedef decltype (gettype<L>()-gettype<R>()) type;};
template<class L, class R> struct MultiplyType { typedef decltype (gettype<L>()*gettype<R>()) type;};
template<class L, class R> struct DivideType   { typedef decltype (gettype<L>()/gettype<R>()) type;};

	

//These are the operations in terms of the a) ACTUAL operation (static function op)
// and, b) The return type (struct Return)
struct Add{
  template<class A, class B, class C>      static A op(const B& b, const C& c){ return b+c; }
  template<class P1, class P2> struct Return { typedef typename AddType<P1,P2>::type Type; };
};

struct Subtract{
  template<class A, class B, class C> static A op(const B& b, const C& c){return b-c;}
  template<class P1, class P2> struct Return { typedef typename SubtractType<P1,P2>::type Type;};
};

struct Multiply{
  template<class A, class B, class C> static A op(const B& b, const C& c){return b*c;}
  template<class P1, class P2> struct Return { typedef typename MultiplyType<P1,P2>::type Type;};
};

// I have added Dot product in order to specialize the operation and use OpenCV's dot product 
// for same-type arguments 
struct DotProduct {
  //template<class A, class B, class C> static A op(const B& b, const C& c){return b*c;}
  template<class P1, class P2> struct Return { typedef typename MultiplyType<P1,P2>::type Type;};
};
	
struct Divide{
  template<class A, class B, class C>   static A op(const B& b, const C& c){return b/c;}
  template<class P1, class P2> struct Return { typedef typename DivideType<P1,P2>::type Type;};
};
	
	  
	






// ********************************** The general template for Operator ****************************
// This is the class (struct) that does the actual computation
// In other words, one layer before operator overloading
template<class Op> struct Operator{};


//////////////////////////////////////////////////////////////////////////////////
//                         Vector <op> Vector
//////////////////////////////////////////////////////////////////////////////////

// ******************* "template constant" for PAIRWISE VECTOR Operator sepcializations ************
// Now this struct IS ESSENTIALLY A TAG for further templating. In effect, it is used as a template "constant".
// It "says" that vector (hence the V in "V"Pairwise) element-wise
// operation involves a) The operation Op, b) The size of the vectors, c) the types P1 and P2
template<typename Op, typename P1, int S, typename P2> struct VElementwise;

// ****************** "template constant" for VECTOR NEGATIONS **********************
// This is easy. It's the vector negation template "constant".
// OpenCV has a negation overload, so it is unenecssary in the context
//template <int S, typename P> struct VNegate; 
	


// Now we need to build Operator classes (actually, structs)
// using the above template "constants" (i.e., vector pairwise operations and vector negation)

// *************** General pairwise Operator specialization ***********************
// This is a vector pairwise general operator (includes addition and multiplication as dot product) 
template<typename Op,typename P1, int S, typename P2> // Here Op is the operation (i.e., Add, Multiply, etc..)
struct Operator<VElementwise<Op, P1, S, P2> > {
	const cv::Vec<P1, S> &left;
	const cv::Vec<P2, S> &right;

	Operator(const cv::Vec<P1, S> &left_in, const cv::Vec<P2, S> &right_in) : left(left_in), right(right_in) {}

	
	typedef typename Op::template Return<P1, P2>::Type P0;
	
	cv::Vec<P0, S> compute() const
	{
		cv::Vec<P0, S> res = cv::Vec<P0, S>();
		for(int i=0; i < left.rows; ++i)
			res[i] = Op::template op<P0, P1, P2>(left[i], right[i]);
		
		return res;
	}
	
};

// The dot product as the only product between cv::Vec 
template<typename P1, int S, typename P2> // Here Op is specialized to DotProduct
struct Operator<VElementwise<DotProduct, P1, S, P2> > {
	const cv::Vec<P1, S> &left;
	const cv::Vec<P2, S> &right;

	Operator(const cv::Vec<P1, S> &left_in, const cv::Vec<P2, S> &right_in) : left(left_in), right(right_in) {}

	typedef typename MultiplyType<P1, P2>::type P0;
	
	P0 compute() const
	{
	  P0 res = 0;
	  for(int i=0; i<left.rows; i++) res += left[i]*right[i];
	
	  return res;
	}
	
};

} // ****************** close MyOperatorOverloads for operator overloads to follow... ******************************

// 1. Addition Vector + Vector 
template<typename P1, int S, typename P2> 
cv::Vec<typename MyOperatorOverloads::AddType<P1, P2>::type, S> operator +(const cv::Vec<P1, S> &v1, const cv::Vec<P2, S> &v2) {
	
  return MyOperatorOverloads::Operator<MyOperatorOverloads::VElementwise<MyOperatorOverloads::Add,P1,S, P2> >(v1,v2).compute();
  
}

// 2. Subtraction Vector - Vector operator
template<int S, typename P1, typename P2> 
cv::Vec<typename MyOperatorOverloads::SubtractType<P1, P2>::type, S> operator -(const cv::Vec<P1, S> &v1, const cv::Vec<P2, S> &v2)
{

  return MyOperatorOverloads::Operator<MyOperatorOverloads::VElementwise<MyOperatorOverloads::Subtract, P1, S, P2> >(v1,v2).compute();
  
}

// 3. diagmult Vector, Vector 
// (George: Rosten refers to elementwise multiplication as matrix multiplication with a diagonal matrix)
template < typename P1, int S, typename P2>
cv::Vec<typename MyOperatorOverloads::MultiplyType<P1,P2>::type, S> diagmult(const cv::Vec<P1,S> &v1, const cv::Vec<P2,S> &v2)
{
	
  return MyOperatorOverloads::Operator<MyOperatorOverloads::VElementwise<MyOperatorOverloads::Multiply, P1, S, P2> >(v1,v2).compute();

}



// Finally, the actual operator overloading for vector-vector multiplication
template<typename P1, int Sz, typename P2>
typename MyOperatorOverloads::MultiplyType<P1, P2>::type operator *(const cv::Vec<P1, Sz> &v1, const cv::Vec<P2, Sz> &v2) {
  
  return MyOperatorOverloads::Operator<MyOperatorOverloads::VElementwise<MyOperatorOverloads::DotProduct, P1, Sz, P2> >(v1, v2).compute();
}

// 6. ^ is the cross product 
template <typename P1, typename P2>
cv::Vec<typename MyOperatorOverloads::MultiplyType<P1,P2>::type, 3> operator ^(const cv::Vec<P1, 3> &v1, const cv::Vec<P2, 3> &v2) {

	// assume the result of adding two restypes is also a restype
	typedef typename MyOperatorOverloads::MultiplyType<P1,P2>::type restype;

	cv::Vec<restype, 3> result;

	
	// [0 -v1(2) v1(1); v1(2) 0 -v1(0); -v1(1) v1(0) 0] * [v2(0); v2(1); v2(2) ]
	
	result[0] =  -v1[2]*v2[1] + v1[1]*v2[2];
	result[1] =   v1[2]*v2[0] - v1[0]*v2[2];
	result[2] =  -v1[1]*v2[0] + v1[0]*v2[1] ;

	return result;
}


// 7. Cross product between vectors stored in cv::Mat as 3x1 matrices
template <typename P1, typename P2>
cv::Mat_<typename MyOperatorOverloads::MultiplyType<P1,P2>::type> operator ^(const cv::Mat_<P1> &mv1, const cv::Mat_<P2> &mv2) {

	// assume the result of adding two restypes is also a restype
	typedef typename MyOperatorOverloads::MultiplyType<P1,P2>::type P0;

	cv::Mat_<P0> result(3, 1);

	
	// [0 -v1(2) v1(1); v1(2) 0 -v1(0); -v1(1) v1(0) 0] * [v2(0); v2(1); v2(2) ]
	
	result(0, 0) =  -mv1(2, 0)*mv2(1, 0) + mv1(1, 0)*mv2(2, 0);
	result(1, 0) =   mv1(2, 0)*mv2(0, 0) - mv1(0, 0)*mv2(2, 0);
	result(2, 0) =  -mv1(1, 0)*mv2(0, 0) + mv1(0, 0)*mv2(1, 0);

	return result;
}



// again opening my operators in order to specialize more Operator templates for matrices
namespace MyOperatorOverloads {

//////////////////////////////////////////////////////////////////////////////////
//                            Matrix <op> Matrix
//////////////////////////////////////////////////////////////////////////////////

// **** Defining the (dummy) template "constants" corresponding to the groups of operations between matrices 
	
// Template "constant" for matrix elementwise operations
template<typename Op,typename P1, typename P2> struct MElementwise;

// Matrix multiplication
template< typename P1,typename P2> struct MatrixMultiply;

// Negation - not necessary with OpenCV matrices
template<typename P> struct MNegate;


// Matrix generic Operator specialized template for any two types of precision (P1, P2)
// for elementwise operations)
template<typename Op,typename P1, typename P2> 
struct Operator<MElementwise<Op, P1, P2> > {

	const cv::Mat_<P1> &left;
	const cv::Mat_<P2> &right;

  
	Operator(const cv::Mat_<P1> &left_in, const cv::Mat_<P2> &right_in) : left(left_in), right(right_in) { }

	typedef typename Op::template Return<P1, P2>::Type P0;
	
	cv::Mat_<P0> compute() const
	{
	    int rrows = min(left.rows, right.rows);
	    int rcols = min(left.cols, right.cols);
	    
	    cv::Mat_<P0> res(rrows, rcols);
		for(int r=0; r < rrows; ++r){
			for(int c=0; c < rcols; ++c){
			  	res(r,c) = Op::template op<P0, P1, P2>(left(r, c), right(r, c));
			}
		}
		return res;
	}
	
};

// Addition Operator struct spcialization for same-type arguments 
// We need to invoke the openCV addfitin operator
template<typename P> struct Operator<MElementwise<Add, P, P> > {

	const cv::Mat_<P> &left;
	const cv::Mat_<P> &right;

  
	Operator(const cv::Mat_<P> &left_in, const cv::Mat_<P> &right_in) : left(left_in), right(right_in) { }

	cv::Mat_<P> compute() const
	{
	    return cv::operator+(left, right);
	}
	
};


// Subtraction Operator struct specialization for same-type arguments (PROBABLY NOT NECESSARY, BUT HEY...)
template<typename P> struct Operator<MElementwise<Subtract, P, P> > {

	const cv::Mat_<P> &left;
	const cv::Mat_<P> &right;

	Operator(const cv::Mat_<P> &left_in, const cv::Mat_<P> &right_in) : left(left_in), right(right_in) { }

	cv::Mat_<P> compute() const
	{
	    return left.sub( right );
	}
	
};

} // ****************** close MyOperatorOverloads for operator overloads to follow... ******************************

// Addition Matrix operator '+' overload
template<typename P1, typename P2> 
cv::Mat_<typename MyOperatorOverloads::AddType<P1, P2>::type> operator +(const cv::Mat_<P1> &m1, const cv::Mat_<P2> &m2) {

  return MyOperatorOverloads::Operator< MyOperatorOverloads::MElementwise<MyOperatorOverloads::Add,P1,P2> >(m1,m2).compute();
}



// Matrix subtraction operator '-' overload
template<typename P1, typename P2> 
cv::Mat_<typename MyOperatorOverloads::SubtractType<P1, P2>::type> operator -(const cv::Mat_<P1> &m1, const cv::Mat_<P2> &m2) {
	
  return MyOperatorOverloads::Operator< MyOperatorOverloads::MElementwise<MyOperatorOverloads::Subtract,P1,P2> >(m1,m2).compute();
  
}



// mult Matrix, Matrix - Elementwise multiplication of matrices
template <typename P1, typename P2>
cv::Mat_<typename MyOperatorOverloads::MultiplyType<P1,P2>::type> mmult(const cv::Mat_<P1> &m1, const cv::Mat_<P2> &m2)
{
	
  return MyOperatorOverloads::Operator<MyOperatorOverloads::MElementwise<MyOperatorOverloads::Multiply,P1,P2 > >(m1,m2).compute();

}




// back to MyOperators again
namespace MyOperatorOverloads {

// Standard Matrix Multiplication Operator specialization for different-type arguments
template<typename P1, typename P2> struct Operator<MatrixMultiply<P1, P2> > {
	const cv::Mat_<P1> &left;
	const cv::Mat_<P2> &right;

	Operator(const cv::Mat_<P1> &left_in, const cv::Mat_<P2> &right_in) : left(left_in), right(right_in) {}

	
	// *********** Non-OpenCV (SLOW, O(n^3) ) multiplication unfortunately.... *******************
	typedef typename MyOperatorOverloads::MultiplyType<P1, P2>::type P0;
	
	cv::Mat_<P0> compute() const {
	  
		cv::Mat_<P0> res(left.rows, right.cols);
		for (int r = 0; r < left.rows; r++)
		  for(int c = 0; c < right.cols; ++c) {
		    P0 sum = 0;
		    for(int j=0; j < right.rows; j++) 
		      sum += left(r, j) * right(j, c);
		    res(r,c) = sum;
		  }
	    return res;
	}
};

// Now Multiplication specialization for same-type arguments 
// Here we invoke the fast, OpenCV multiplication operator
template<typename P> struct Operator<MatrixMultiply<P, P> > {
	const cv::Mat_<P> &left;
	const cv::Mat_<P> &right;

	Operator(const cv::Mat_<P> &left_in, const cv::Mat_<P> &right_in) : left(left_in), right(right_in) {}

	
	// ***********Now, can use OpenCV's fast multiplication *******************
	cv::Mat_<P> compute() const
	{
	  return cv::operator*(left, right); 

	}
	
};

} // ****************** close MyOperatorOverloads for operator overloads to follow... ******************************

// Standard Matrix multiplication Matrix * Matrix : The actual operator
template<typename P1, typename P2> 
cv::Mat_<typename MyOperatorOverloads::MultiplyType<P1, P2>::type> operator *(const cv::Mat_<P1> &m1, const cv::Mat_<P2> &m2) {
	
  return MyOperatorOverloads::Operator<MyOperatorOverloads::MatrixMultiply<P1, P2> >(m1,m2).compute();
}



// again back to myOperaotrs...
namespace MyOperatorOverloads {
//////////////////////////////////////////////////////////////////////////////////
//                 matrix <op> vector and vv.
//////////////////////////////////////////////////////////////////////////////////



// Template "constant" struct for Matrix * vector
template<typename P1, typename P2, int Sz> struct MatrixVectorMultiply;

  
//Tenplate "constant" for vector * Matrix (in the v.t() * M fashion)
template<typename P1, int Sz, typename P2> struct VectorMatrixMultiply;

	
// Template "constant" struct for Matrix * Vector DIAGONAL multiply (we'll see what the hell is this...)
// I am guessing Rosten means .* columwise
template<typename P1, typename P2, int Sz> struct MatrixVectorDiagMultiply;




// Matrix-Vector standard multiplication Matrix * Vector DIFFERENT-TYPES 
// The return type is cv::Mat_. for obvious reasons. I don't want to use size variables in the templates.
// To resolve this i will add a conversion method from cv::Mat to cv::Vec
template<typename P1, typename P2, int Sz> 
struct Operator<MatrixVectorMultiply<P1, P2, Sz> > {
	const cv::Mat_<P1> &M;
	const cv::Vec<P2, Sz> &v;

	Operator(const cv::Mat_<P1> &M_in, const cv::Vec<P2, Sz> &v_in) : M(M_in), v(v_in) {}

	typedef typename MultiplyType<P1, P2>::type P0;
	
	cv::Mat_<P0> compute() const {
	  
	  cv::Mat_<P0> res(M.rows, 1); // we construct a rowsx1 matrix (it should suffice
	
	  for(int i=0; i < M.rows; ++i) {
	    res(i, 0) = 0;
	    for (int j = 0; j < v.rows; j++) 
		res(i, 0) += Multiply::template op<P0, P1, P2>(M(i, j) , v[j]); 
	  }
	  
	  return res;
	}
};

} // ****************** close MyOperatorOverloads for operator overloads to follow... ******************************


// and the Matrix-vector multiplication operator overload!
template<typename PM, typename PV, int Sz>
cv::Mat_<typename MyOperatorOverloads::MultiplyType<PM, PV>::type> operator *(const cv::Mat_<PM> &m, const cv::Vec<PV, Sz> &v)
{
	return MyOperatorOverloads::Operator<MyOperatorOverloads::MatrixVectorMultiply<PM, PV, Sz> >(m,v).compute();
}
																	


// Now, it is easy to get the Vector * Matrix operator...
template<typename PV, int Sz, typename PM> 
cv::Mat_<typename MyOperatorOverloads::MultiplyType<PV, PM>::type > operator *(const cv::Vec<PV, Sz> &v, const cv::Mat_<PM> &m) {
	
	//cv::Mat_<P1> mt(m.t());
	
	return (m.t() * v).t();
}



																	




#endif