// *-* c++ *-*
// Copyright 2008 Isis Innovation Limited

// N-th implementation of a camera model
// GK 2007
// Evolved a half dozen times from the CVD-like model I was given by
// TWD in 2000
// 
// This one uses the ``FOV'' distortion model of
// Deverneay and Faugeras, Straight lines have to be straight, 2001
//
// BEWARE: This camera model caches intermediate results in member variables
// Some functions therefore depend on being called in order: i.e.
// GetProjectionDerivs() uses data stored from the last Project() or UnProject()
// THIS MEANS YOU MUST BE CAREFUL WITH MULTIPLE THREADS
// Best bet is to give each thread its own version of the camera!
//
// Camera parameters are stored in a GVar, but changing the gvar has no effect
// until the next call to RefreshParams() or SetImageSize().
//
// Pixel conventions are as follows:
// For Project() and Unproject(),
// round pixel values - i.e. (0.0, 0.0) - refer to pixel centers
// I.e. the top left pixel in the image covers is centered on (0,0)
// and covers the area (-.5, -.5) to (.5, .5)
//
// Be aware that this is not the same as what opengl uses but makes sense
// for acessing pixels using ImageRef, especially ir_rounded.
//
// What is the UFB?
// This is for projecting the visible image area
// to a unit square coordinate system, with the top-left at 0,0,
// and the bottom-right at 1,1
// This is useful for rendering into textures! The top-left pixel is NOT
// centered at 0,0, rather the top-left corner of the top-left pixel is at 
// 0,0!!! This is the way OpenGL thinks of pixel coords.
// There's the Linear and the Distorting version - 
// For the linear version, can use 
// glMatrixMode(GL_PROJECTION); glLoadIdentity();
// glMultMatrix(Camera.MakeUFBLinearFrustumMatrix(near,far));
// To render un-distorted geometry with full frame coverage.
//

#ifndef __ATAN_CAMERA_H
#define __ATAN_CAMERA_H


#define DEFAULT_IMG_HEIGHT 480
#define DEFAULT_IMG_WIDTH 640


#include <cmath>
#include "Persistence/PVars.h"


#include "OpenCV.h"

#define NUMTRACKERCAMPARAMETERS 5


class CameraCalibrator;
class CalibImage;

// The parameters are:
// 0 - normalized x focal length
// 1 - normalized y focal length
// 2 - normalized x offset
// 3 - normalized y offset
// 4 - w (distortion parameter)

class ATANCamera {
 public:
   // Default camera parameters
  static cv::Vec<float, NUMTRACKERCAMPARAMETERS> mvDefaultParams;
  
  ATANCamera(std::string sName, const cv::Size2i imgsize = cv::Size2i(DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT) );
  
  // Image size get/set: updates the internal projection params to that target image size.
  void SetImageSize(const cv::Vec2f &v2ImageSize); // this is inline
  void SetImageSize(const cv::Size2i &imSize);     // the overloadf CANNOT be..
  
  cv::Vec2f GetImageSize() {return mvImageSize;};
  void RefreshParams();
  
  // Various projection functions
  cv::Vec2f Project(const cv::Vec2f&); // Projects from camera z=1 plane to pixel coordinates, with radial distortion
  cv::Vec2f UnProject(const cv::Vec2f&); // Inverse operation
  
  cv::Vec2f UFBProject(const cv::Vec2f &camframe);
  cv::Vec2f UFBUnProject(const cv::Vec2f &camframe);
  inline cv::Vec2f UFBLinearProject(const cv::Vec2f &camframe);
  inline cv::Vec2f UFBLinearUnProject(const cv::Vec2f &fbframe);
  
  cv::Mat_<float> GetProjectionDerivs(); // 2x2 Projection jacobian
  
  inline bool Invalid() {  return mbInvalid;}
  inline double LargestRadiusInImage() {  return mdLargestRadius; }
  inline double OnePixelDist() { return mdOnePixelDist; }
  
  // The z=1 plane bounding box of what the camera can see
  inline cv::Vec2f ImplaneTL(); 
  inline cv::Vec2f ImplaneBR(); 

  // OpenGL helper function
  cv::Mat_<float> MakeUFBLinearFrustumMatrix(float near, float far); // Returns A 4x4 matrix
  
  // Feedback for Camera Calibrator
  double PixelAspectRatio() { return mvFocal[1] / mvFocal[0];}
  
  
  
  
  
 protected:
  Persistence::pvar3<cv::Vec<float, NUMTRACKERCAMPARAMETERS> > mpvvCameraParams; // The actual camera parameters
  
  
  cv::Mat_<float> GetCameraParameterDerivs(); // 2x NUMTRACKERCAMPARAMETERS
  void UpdateParams(cv::Vec<float, NUMTRACKERCAMPARAMETERS> vUpdate);
  void DisableRadialDistortion();
  
  // Cached from the last project/unproject:
  cv::Vec2f mvLastCam;      // Last z=1 coord
  cv::Vec2f mvLastIm;       // Last image/UFB coord
  cv::Vec2f mvLastDistCam;  // Last distorted z=1 coord
  double mdLastR;           // Last z=1 radius
  double mdLastDistR;       // Last z=1 distorted radius
  double mdLastFactor;      // Last ratio of z=1 radii
  bool mbInvalid;           // Was the last projection invalid?
  
  // Cached from last RefreshParams:
  float mdLargestRadius; // Largest R in the image
  float mdMaxR;          // Largest R for which we consider projection valid
  float mdOnePixelDist;  // z=1 distance covered by a single pixel offset (a rough estimate!)
  float md2Tan;          // distortion model coeff
  float mdOneOver2Tan;   // distortion model coeff
  float mdW;             // distortion model coeff
  float mdWinv;          // distortion model coeff
  float mdDistortionEnabled; // One or zero depending on if distortion is on or off.
  cv::Vec2f mvCenter;     // Pixel projection center
  cv::Vec2f mvFocal;      // Pixel focal length
  cv::Vec2f mvInvFocal;   // Inverse pixel focal length
  cv::Vec2f mvImageSize;  
  cv::Vec2f mvUFBLinearFocal;
  cv::Vec2f mvUFBLinearInvFocal;
  cv::Vec2f mvUFBLinearCenter;
  cv::Vec2f mvImplaneTL;   
  cv::Vec2f mvImplaneBR;
  
  // Radial distortion transformation factor: returns ratio of distorted / undistorted radius.
  // George: This IS the correction factor in the projection model: 
  // You need to multiply the Euclidean normalized coordinates by this factor BEFORE you project them onto the image
  // reason being, we have to "distort" the coordinates before we send them to the image
  /// Returns the distorted radius on the normalized Euclidean plane divided by the undistorted one
  /// This factor can be used verbatoc for projection to the image
  inline float rtrans_factor(float r)
  {
    if(r < 0.001 || mdW == 0.0) return 1.0;
    else 
      return (mdWinv* atan(r * md2Tan) / r); // 1/w * atan(2*ru*tan(w/2)) / ru 
  };

  // Inverse radial distortion: returns un-distorted radius from distorted.
  inline float invrtrans(float r)
  {
    if(mdW == 0.0) return r;
    return(tan(r * mdW) * mdOneOver2Tan); // mdOneOver2Tan is a radial distortion coefficient (see the paper by Devernay - Faugeras)
  };
  
  std::string msName;

  friend class CameraCalibrator;   // friend declarations allow access to calibration jacobian and camera update function.
  friend class CalibImage;
};

// Some inline projection functions:
inline cv::Vec2f ATANCamera::UFBLinearProject(const cv::Vec2f &camframe)
{
  cv::Vec2f v2Res;
  v2Res[0] = camframe[0] * mvUFBLinearFocal[0] + mvUFBLinearCenter[0];
  v2Res[1] = camframe[1] * mvUFBLinearFocal[1] + mvUFBLinearCenter[1];
  return v2Res;
}

inline cv::Vec2f ATANCamera::UFBLinearUnProject(const cv::Vec2f &fbframe)
{
  cv::Vec2f v2Res;
  v2Res[0] = (fbframe[0] - mvUFBLinearCenter[0]) * mvUFBLinearInvFocal[0];
  v2Res[1] = (fbframe[1] - mvUFBLinearCenter[1]) * mvUFBLinearInvFocal[1];
  return v2Res;
}


#endif

