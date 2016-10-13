// George Terzakis 2016 - Tracking and mapping with OpenCV - GTAM Calibrator
// Based on PTAM by Klein and Murray

#include "OpenGL.h"
#include "Persistence/instances.h"

#include "CameraCalibrator.h"
#include "ATANCamera.h"

#include <fstream>
#include <stdlib.h>

#include "GCVD/GLHelpers.h"



using namespace std;
using namespace Persistence;




int main()
{
  cout << "  Welcome to the George's CameraCalibrator for Tracking and Mapping" << endl;
  cout << "  ----------------------------------------------------------------- " << endl;

  cout << "  A deep code adaptation of the original 'Parallel tracking and Mapping' by Klein and Murray 2008" << endl;
  cout << "  This code is purely OpenCV based and DOES NOT REQUIRE libCVD, TooN and GVars (although you will need openGL)" << endl;
  cout << " *"<<endl;
  cout << " Partial functionality of TooN, libCVD and GVars has been retained/adapted based on the latest code by Rosten and Klein"<<endl;
  cout << " for interoperability with OpenCV methods and structures."<<endl;
  cout << endl;
  cout << " ***************** George Terzakis 2016 *********************" <<endl;
  cout << " **************** University of Portsmouth ******************" <<endl;
  cout << endl;  
  cout << "  Parsing calibrator_settings.cfg ...." << endl;
  
  GUI.LoadFile("calibrator_settings.cfg");

  GUI.StartParserThread();
  atexit(GUI.StopParserThread); // Clean up readline when program quits
  
  
  try
    {
      CameraCalibrator c;
      
      c.Run();
    }
    catch(cv::Exception e)
    {
      cout << endl;
      cout << "!! Failed to run CameraCalibrator; got exception. " << endl;
      cout << "   Exception was: " << endl;
      //cout << e.what << endl;
      cout <<"At line : " << e.line << endl << e.msg << endl;
    }
}







CameraCalibrator::CameraCalibrator() : mGLWindow(mVideoSource.getSize(), "Camera Calibrator"), mCamera("Camera", mVideoSource.getSize())
{
  
  
  mbDone = false;
  
  
  GUI.RegisterCommand("CameraCalibrator.GrabNextFrame", GUICommandCallBack, this);
  GUI.RegisterCommand("CameraCalibrator.Reset", GUICommandCallBack, this);
  GUI.RegisterCommand("CameraCalibrator.ShowNext", GUICommandCallBack, this);
  GUI.RegisterCommand("CameraCalibrator.SaveCalib", GUICommandCallBack, this);
  GUI.RegisterCommand("quit", GUICommandCallBack, this);
  GUI.RegisterCommand("exit", GUICommandCallBack, this);
  
  PV3::Register(mpvnOptimizing, "CameraCalibrator.Optimize", 0, SILENT);
  PV3::Register(mpvnShowImage, "CameraCalibrator.Show", 0, SILENT);
  PV3::Register(mpvnDisableDistortion, "CameraCalibrator.NoDistortion", 1, SILENT);
    
  GUI.ParseLine("GLWindow.AddMenu CalibMenu");
  GUI.ParseLine("CalibMenu.AddMenuButton Live GrabFrame CameraCalibrator.GrabNextFrame");
  GUI.ParseLine("CalibMenu.AddMenuButton Live Reset CameraCalibrator.Reset");
  GUI.ParseLine("CalibMenu.AddMenuButton Live Optimize \"CameraCalibrator.Optimize=1\"");
  GUI.ParseLine("CalibMenu.AddMenuToggle Live NoDist CameraCalibrator.NoDistortion");
  GUI.ParseLine("CalibMenu.AddMenuSlider Opti \"Show Img\" CameraCalibrator.Show 0 10");
  GUI.ParseLine("CalibMenu.AddMenuButton Opti \"Show Next\" CameraCalibrator.ShowNext");
  GUI.ParseLine("CalibMenu.AddMenuButton Opti \"Grab More\" CameraCalibrator.Optimize=0 ");
  GUI.ParseLine("CalibMenu.AddMenuButton Opti Reset CameraCalibrator.Reset");
  GUI.ParseLine("CalibMenu.AddMenuToggle Opti NoDist CameraCalibrator.NoDistortion");
  GUI.ParseLine("CalibMenu.AddMenuButton Opti Save CameraCalibrator.SaveCalib");
  Reset();
  
  
  
  cout << " Initial camera parameters : " << *mCamera.mpvvCameraParams <<endl;						 
  cout << " Default camera parameters : " << ATANCamera::mvDefaultParams <<endl;					 
  cout << "Image size as provided by the VideoSource object: " << mCamera.GetImageSize()[0]<<" , "<<mCamera.GetImageSize()[1]<<endl;
 
}

void CameraCalibrator::Run()
{
  while(!mbDone) {
    
      // We use two versions of each video frame:
      // One black and white (for processing by the tracker etc)
      // and one RGB, for drawing.
      
      cv::Mat imFrameRGB;
      cv::Mat_<uchar> imFrameBW;
      
      // Grab new video frame...
      mVideoSource.GetAndFillFrameBWandRGB(imFrameBW, imFrameRGB);  
      
      
      // Set up openGL. more comments in the following methods in GLWindow.h ...
      mGLWindow.SetupViewport();
      mGLWindow.SetupVideoOrtho();
      mGLWindow.SetupVideoRasterPosAndZoom();
       
      
      // Note here that a "CalibImage" here represents an object that contains ALL the information necessary 
      // for camera parameter optimization (i.e., corner locations arranged in a grid).
      // Thus, the "mpvnOptimizing" flag - if true - implies that we can run optimization over the camera parameters  
      if(mvCalibImgs.size() < 1) *mpvnOptimizing = 0; // if no calibration images exist, then set the optimization flag to false; 
      
      if(!*mpvnOptimizing) {
	
	  GUI.ParseLine("CalibMenu.ShowMenu Live");
	  
    
	  // draw the grayscale image on the OpenGL canvas
	  GLXInterface::glDrawPixelsGRAY(imFrameBW);
	  //GLXInterface::glDrawPixelsBGR(imFrameRGB); 

	  // create a Calibration image
	  CalibImage c;
	  // The method "MakeFromImage" does it all: 
	  // a) Detect free lying corners and display them as red dots.
	  // b) Pick a starting free corner and find its pose (parameters).
	  // c) detect more corners arranged in a rectangular grid using the above starting corner.
	  // d) Draw the grid.
	  // If true, "MakeFromImage" has actually found a number of grid corners connected to each other 
	  // and therefore can be used to optimize camera parameters.
	  if(c.MakeFromImage(imFrameBW, imFrameRGB) ) {
	      // if a frame capture was requested (frame grabbing here means, "REGISTER A GOOD CALIBRATION IMAGE" 
	      // and NOT raw frame capturing as the name of the variable or the menu caption implies)
	      if(mbGrabNextFrame)
		{
		  // keep the calibration image in the list
		  mvCalibImgs.push_back(c);
		 // Now work out an initial impression of camera pose from the calibration image
		  mvCalibImgs.back().GuessInitialPose(mCamera);
		  
		  // draw a cool 3D projection grid
// 		  mvCalibImgs.back().Draw3DGrid(mCamera, false);
		  // switch back to waiting for the user to request the capture of a good caibration image
		  mbGrabNextFrame = false;
		  
		  
		};
	    
	    cout << "Image was 'made'"<<endl;
	    
	   }
	    
      }
      else {
	  
	   //cout << "Optimizing..."<<endl;
	
	  OptimizeOneStep();
      
	  GUI.ParseLine("CalibMenu.ShowMenu Opti");
	  int nToShow = *mpvnShowImage - 1;
	  
	  if(nToShow < 0) nToShow = 0;
	  if(nToShow >= (int) mvCalibImgs.size())  nToShow = mvCalibImgs.size()-1;
	  
	  *mpvnShowImage = nToShow + 1;
      
	  GLXInterface::glDrawPixelsGRAY(mvCalibImgs[nToShow].mim);
	  
	  mvCalibImgs[nToShow].Draw3DGrid(mCamera,true);
	}
	
      
      ostringstream ost;
      ost << "Camera Calibration: Grabbed " << mvCalibImgs.size() << " images." << endl;
      if(!*mpvnOptimizing)
	{
	  ost << "Take snapshots of the calib grid with the \"GrabFrame\" button," << endl;
	  ost << "and then press \"Optimize\"." << endl;
	  ost << "Take enough shots (4+) at different angles to get points " << endl;
	  ost << "into all parts of the image (corners too.) The whole grid " << endl;
	  ost << "doesn't need to be visible so feel free to zoom in." << endl;
	}
      else
	{
	  ost << "Current RMS pixel error is " << mdMeanPixelError << endl;
	  //ost << "Current camera params are  " << PV3::get_var("Camera.Parameters") << endl;
	  ost << "Current camera params are  " << *mCamera.mpvvCameraParams << endl;
	  ost << "(That would be a pixel aspect ratio of " 
	      <<  mCamera.PixelAspectRatio() << ")" << endl;
	  ost << "Check fit by looking through the grabbed images." << endl;
	  ost << "RMS should go below 0.5, typically below 0.3 for a wide lens." << endl;
	  ost << "Press \"save\" to save calibration to camera.cfg file and exit." << endl;
	}

      mGLWindow.DrawCaption(ost.str());
      mGLWindow.DrawMenus();
      mGLWindow.HandlePendingEvents();
      mGLWindow.swap_buffers();
    }
}

void CameraCalibrator::Reset()
{
  
  PV3::get<cv::Vec<float, NUMTRACKERCAMPARAMETERS> >("Camera.Parameters", ATANCamera::mvDefaultParams, SILENT);
  
  
  if(*mpvnDisableDistortion) mCamera.DisableRadialDistortion();
  
  mCamera.SetImageSize(mVideoSource.getSize());
  mbGrabNextFrame =false;
  *mpvnOptimizing = false;
  mvCalibImgs.clear();
}

void CameraCalibrator::GUICommandCallBack(void* ptr, string sCommand, string sParams)
{
  ((CameraCalibrator*) ptr)->GUICommandHandler(sCommand, sParams);
}

void CameraCalibrator::GUICommandHandler(string sCommand, string sParams)  // Called by the callback func..
{
  if(sCommand=="CameraCalibrator.Reset")
    {
      Reset();
      return;
    };
  if(sCommand=="CameraCalibrator.GrabNextFrame")
    {
      mbGrabNextFrame = true;
      cout << "Can I please grab the next frame????" <<endl;
      return;
    }
  if(sCommand=="CameraCalibrator.ShowNext")
    {
      int nToShow = (*mpvnShowImage - 1 + 1) % mvCalibImgs.size();
      *mpvnShowImage = nToShow + 1;
      return;
    }
  if(sCommand=="CameraCalibrator.SaveCalib")
    {
      cout << "  Camera calib is " << PV3::get_var("Camera.Parameters") << endl;
      cout << "  Saving camera calib to camera.cfg..." << endl;
      ofstream ofs("camera.cfg");
      if(ofs.good())
	{
	  
	  PV3::PrintVar("Camera.Parameters", ofs);
	  
	  ofs.close();
	  cout << "  .. saved."<< endl;
	}
      else
	{
	  cout <<"! Could not open camera.cfg for writing." << endl;
	  PV3.PrintVar("Camera.Parameters", cout);
	  cout <<"  Copy-paste above line to settings.cfg or camera.cfg! " << endl;
	}
      mbDone = true;
    }
  if(sCommand=="exit" || sCommand=="quit")
    {
      mbDone = true;
    }
}


// Optimize camera parameters using the list of selected calibratin images
void CameraCalibrator::OptimizeOneStep()
{
  
  int nViews = mvCalibImgs.size();
  int nDim = 6 * nViews + NUMTRACKERCAMPARAMETERS;
  int nCamParamBase = nDim - NUMTRACKERCAMPARAMETERS;
  
  // preparing LS
  // The information matrix
  cv::Mat_<double> mJTJ = cv::Mat_<double>::eye(nDim, nDim);
  // information vector
  cv::Mat_<double> vJTe = cv::Mat_<double>::zeros(nDim, 1); // a matrix vector... Smells like Least Squares....
  
  if(*mpvnDisableDistortion) mCamera.DisableRadialDistortion();

  // sum of squared errors
  double dSumSquaredError = 0.0;
  int nTotalMeas = 0;
  
  cv::Mat_<double> mv2Error(2, 1); // temporary storage for v2error as a 2x1 matrix (to accommodate multiplications)
  
  // For consistency and potential error checking, I am retaining old PTAM code
  for(int n=0; n<nViews; n++) {
    
      int nMotionBase = n*6;
      vector<CalibImage::ErrorAndJacobians> vEAJ = mvCalibImgs[n].Project(mCamera);
  
      if (vEAJ.size() == 0 ) {
	cout << "All point projections are invalid with current parameters. Leaving image out of the optimization..."<<endl;
	
	continue;
	
      }
  
      for(unsigned int i=0; i<vEAJ.size(); i++) {

	  CalibImage::ErrorAndJacobians &EAJ = vEAJ[i];
	  // All the below should be +=, but the MSVC compiler doesn't seem to understand that. :( George: We'll have to see about this...
	  //mJTJ.slice(nMotionBase, nMotionBase, 6, 6) = 
	  //mJTJ.slice(nMotionBase, nMotionBase, 6, 6) + EAJ.m26PoseJac.T() * EAJ.m26PoseJac; // tricky one...
	  cv::Mat_<double> mJTJblock6x6 = mJTJ( cv::Range(nMotionBase, nMotionBase + 6), cv::Range(nMotionBase, nMotionBase + 6) );
	  cv::Mat_<double> tempBlock6x6 = mJTJblock6x6 + EAJ.m26PoseJac.t() * EAJ.m26PoseJac;
	  tempBlock6x6.copyTo(mJTJblock6x6);
	  
	 
	  
	  //mJTJ.slice(nCamParamBase, nCamParamBase, NUMTRACKERCAMPARAMETERS, NUMTRACKERCAMPARAMETERS) = 
	  //mJTJ.slice(nCamParamBase, nCamParamBase, NUMTRACKERCAMPARAMETERS, NUMTRACKERCAMPARAMETERS) + EAJ.m2NCameraJac.T() * EAJ.m2NCameraJac;
	  cv::Mat_<double> mJTJBlocknxn = mJTJ( cv::Range(nCamParamBase, nCamParamBase + NUMTRACKERCAMPARAMETERS), 
						cv::Range(nCamParamBase, nCamParamBase + NUMTRACKERCAMPARAMETERS) );
	  cv::Mat_<double> tempBlocknxn = mJTJBlocknxn + EAJ.m2NCameraJac.t() * EAJ.m2NCameraJac;
	  tempBlocknxn.copyTo(mJTJBlocknxn);
	  
	  //mJTJ.slice(nMotionBase, nCamParamBase, 6, NUMTRACKERCAMPARAMETERS) =
	  //mJTJ.slice(nMotionBase, nCamParamBase, 6, NUMTRACKERCAMPARAMETERS) + EAJ.m26PoseJac.T() * EAJ.m2NCameraJac;
	  cv::Mat_<double> mJTJBlock6xn = mJTJ( cv::Range(nMotionBase, nMotionBase + 6), 
						cv::Range(nCamParamBase, nCamParamBase + NUMTRACKERCAMPARAMETERS) );
	  cv::Mat_<double> tempBlock6xn = mJTJBlock6xn + EAJ.m26PoseJac.t() * EAJ.m2NCameraJac;
	  tempBlock6xn.copyTo(mJTJBlock6xn);
	  
	  
	  //mJTJ.T().slice(nMotionBase, nCamParamBase, 6, NUMTRACKERCAMPARAMETERS) = 
	  //mJTJ.T().slice(nMotionBase, nCamParamBase, 6, NUMTRACKERCAMPARAMETERS) + EAJ.m26PoseJac.T() * EAJ.m2NCameraJac;
	  cv::Mat_<double> mJTJBlocknx6 = mJTJ( cv::Range(nCamParamBase, nCamParamBase + NUMTRACKERCAMPARAMETERS), 
						cv::Range(nMotionBase, nMotionBase + 6) );
	  cv::Mat_<double> tempBlocknx6 = tempBlock6xn.t();
	  tempBlocknx6.copyTo(mJTJBlocknx6);
	  
	  
	  // Above does twice the work it needs to, but who cares.. (George: Maybe a bit faster; now... But still slow I guess...)
	  //vJTe.slice(nMotionBase,6) = 
	  //vJTe.slice(nMotionBase,6) + EAJ.m26PoseJac.T() * EAJ.v2Error;
	  
	  mv2Error(0, 0) = EAJ.v2Error[0];
	  mv2Error(1, 0) = EAJ.v2Error[1];
	  
	  cv::Mat_<double> vJTe6 = vJTe(cv::Range(nMotionBase, nMotionBase + 6), cv::Range::all() );
	  cv::Mat_<double> tempv6 = vJTe6 + EAJ.m26PoseJac.t() * mv2Error;
	  tempv6.copyTo(vJTe6);
	  
	  //vJTe.slice(nCamParamBase,NUMTRACKERCAMPARAMETERS) = 
	  //vJTe.slice(nCamParamBase,NUMTRACKERCAMPARAMETERS) + EAJ.m2NCameraJac.T() * EAJ.v2Error;

	  cv::Mat_<double> vJTen = vJTe(cv::Range(nCamParamBase, nCamParamBase + NUMTRACKERCAMPARAMETERS), cv::Range::all() );
	  
	  cv::Mat_<double> tempvn = vJTen + EAJ.m2NCameraJac.t() * mv2Error;
	  tempvn.copyTo(vJTen);
	  
	  //dSumSquaredError += EAJ.v2Error * EAJ.v2Error;
	  dSumSquaredError += EAJ.v2Error[0] * EAJ.v2Error[0] + EAJ.v2Error[1] * EAJ.v2Error[1];
	 
	  
	  
	  ++nTotalMeas;
	}
    };
  
  if (nTotalMeas == 0) {
    cout << "Did not manage to include a single grid corner in the optimization ! Skipping updates !" <<endl;
    return;
  }
    
  mdMeanPixelError = sqrt(dSumSquaredError / nTotalMeas);
  
  
	  
  cv::Mat_<double> vUpdate(nDim, 1);
  cv::solve(mJTJ, vJTe, vUpdate, cv::DECOMP_CHOLESKY);
  vUpdate *= 0.1; // Slow down because highly nonlinear...
  for(int n=0; n<nViews; n++) {
    cv::Mat_<double> vUslice = vUpdate(cv::Range(n*6, n*6 + 6), cv::Range::all() );
    //mvCalibImgs[n].mse3CamFromWorld = SE3<>::exp(vUpdate.slice(n * 6, 6)) * mvCalibImgs[n].mse3CamFromWorld;
    SE3<> Dse3 = SE3<>::exp( cv::Vec<float, 6>( vUslice(0, 0), 
					      vUslice(1, 0), 
					      vUslice(2, 0), 
					      vUslice(3, 0), 
					      vUslice(4, 0), 
					      vUslice(5, 0) )
			  );
    mvCalibImgs[n].mse3CamFromWorld = Dse3 * mvCalibImgs[n].mse3CamFromWorld; 
						
   
  }
  //mCamera.UpdateParams(vUpdate.slice(nCamParamBase, NUMTRACKERCAMPARAMETERS));
  cv::Vec<float, NUMTRACKERCAMPARAMETERS> Dparams;
  for (int k = 0; k<NUMTRACKERCAMPARAMETERS; k++) Dparams[k] = vUpdate(nCamParamBase+k, 0);
 
  mCamera.UpdateParams(Dparams);
};














