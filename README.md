A deep modification of the PTAM Calibrator implemented using OpenCV ONLY (i.e., no TooN, no libCVD and no GVars).


This is a deep source modification of the original calibration application of the "Parallel Tracking and Mapping" (PTAM) by Klein and Murray to work exclusively with OpenCV and OpenGL. Libraries TooN, liCVD and GVars have been removed or partially modified into new code that provides the respective functionality to the calibrator code (which also has been modified to a certain extent ).

The calibrator itself has changed, but in principle is the same. Minor improvements where made in the way the first grid corner is detected and initial camera pose as well as corner angle guessing; additional parameters were introduced in the configuration file to accommodate tuning, primarily in the cases of cheap cameras. I have added plenty of comments in order for everyone to be able to hack the code at any stage.

I will be shortly adding a generic camera model (i.e., standard pinhole with polynomial radial distortion model).

The original PTAM file names were kept. Additional code is organized in the following directories:

a) GCVD: Basically some libCVD and TooN functionality, including new operator overloads (ideas here are loosely based on the brilliant ways that TooN operators were setup, but mostly new stuff due to the gap between OpenCV "Mat" and "Vec" objects as opposed to TooN's matrices and vectors ). In short, operations between matrices and vectors of different primitives are possible (as in TooN), but with openCV materials... The directory also contains code for OpenGL based interface (modifications on Klein and Rosten's code).

b) Persistence: The code here provides functionality almost identical to the one by GVars. Of course, now OpenCV vectors and matrices can be persistent (loosely replacing the TooN stuff). The GUI class is practically a subset of the original GUI in GVars.

c) FAST: Some FAST headers lifted almost verbatim and thereafter adapted to work with OpenCV matrices (images); basically original code with many simple hacks...
