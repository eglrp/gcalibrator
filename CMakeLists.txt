cmake_minimum_required(VERSION 2.6)
set(PROJ_NAME gcalibrator)

##############################################
## External libraries


######################################
# The external reference directory (searching) for cmake
set(EXT_DIR ${CMAKE_SOURCE_DIR}/cmake )
list( APPEND CMAKE_MODULE_PATH
  ${EXT_DIR}
)

## find openCV
set( OpenCV_INCLUDE_DIRS
	"/usr/local/include/opencv"
	"/usr/local/include/opencv2"
)

set( OpenCV_LIBS
	"/usr/local/lib/libopencv_calib3d.so"
	"/usr/local/lib/libopencv_core.so"
	"/usr/local/lib/libopencv_features2d.so"
	"/usr/local/lib/libopencv_flann.so"
	"/usr/local/lib/libopencv_highgui.so"
	#"/usr/local/lib/libopencv_imgcodecs.so"
	"/usr/local/lib/libopencv_imgproc.so"
	"/usr/local/lib/libopencv_ml.so"
	"/usr/local/lib/libopencv_objdetect.so"
	"/usr/local/lib/libopencv_photo.so"
	#"/usr/local/lib/libopencv_shape.so"
	"/usr/local/lib/libopencv_stitching.so"
	"/usr/local/lib/libopencv_superres.so"
	"/usr/local/lib/libopencv_video.so"
	#"/usr/local/lib/libopencv_videoio.so"
	"/usr/local/lib/libopencv_videostab.so"
	
)


list( APPEND
	EXT_INCLUDE_DIRS
	${OpenCV_INCLUDE_DIRS}
)
list( APPEND
	EXT_LIBS
	${OpenCV_LIBS}
)

#find_package(LAPACK REQUIRED)
#if(LAPACK_FOUND)
#set(LAPACKLIBS_FOUND TRUE)
#list( APPEND
#	EXT_LINKER_FLAGS
#	${LAPACK_LINKER_FLAGS}
#)
#list( APPEND
#	EXT_LIBS
#	${LAPACK_LIBRARIES}
#	gfortran
#)
#endif()

## find OpenGL
find_package(OPENGL REQUIRED)
list( APPEND
	EXT_INCLUDE_DIRS
	${OPENGL_INCLUDE_DIR}
)
list( APPEND
	EXT_LIBS
	${OPENGL_LIBRARIES}
)


########## PROJ_NAME SOURCE FILES  ###################
	

  
set(PROJ_SOURCE
	${CMAKE_SOURCE_DIR}/CameraCalibrator.cpp
	${CMAKE_SOURCE_DIR}/GLWindow2.cpp	
	${CMAKE_SOURCE_DIR}/GLWindowMenu.cpp
	${CMAKE_SOURCE_DIR}/VideoSource.cpp
	${CMAKE_SOURCE_DIR}/CalibImage.cpp
	${CMAKE_SOURCE_DIR}/CalibCornerPatch.cpp
	${CMAKE_SOURCE_DIR}/ATANCamera.cpp
	${CMAKE_SOURCE_DIR}/FAST/fast_7_detect.cpp
	${CMAKE_SOURCE_DIR}/FAST/fast_7_score.cpp
	${CMAKE_SOURCE_DIR}/FAST/fast_8_detect.cpp
	${CMAKE_SOURCE_DIR}/FAST/fast_8_score.cpp
	${CMAKE_SOURCE_DIR}/FAST/fast_9_detect.cpp
	${CMAKE_SOURCE_DIR}/FAST/fast_9_score.cpp
        ${CMAKE_SOURCE_DIR}/FAST/fast_10_detect.cpp
	${CMAKE_SOURCE_DIR}/FAST/fast_10_score.cpp
	${CMAKE_SOURCE_DIR}/FAST/fast_11_detect.cpp
	${CMAKE_SOURCE_DIR}/FAST/fast_11_score.cpp
	${CMAKE_SOURCE_DIR}/FAST/fast_12_detect.cpp
	${CMAKE_SOURCE_DIR}/FAST/fast_12_score.cpp
	${CMAKE_SOURCE_DIR}/FAST/nonmax_suppression.cpp
	${CMAKE_SOURCE_DIR}/FAST/fast_corner.cpp
	${CMAKE_SOURCE_DIR}/GCVD/GLWindow.cpp
	${CMAKE_SOURCE_DIR}/GCVD/GLText.cpp
	${CMAKE_SOURCE_DIR}/Persistence/PVars.cpp
	${CMAKE_SOURCE_DIR}/Persistence/instances.cpp
	${CMAKE_SOURCE_DIR}/Persistence/GUI.cpp
	${CMAKE_SOURCE_DIR}/Persistence/GUI_language.cpp
	${CMAKE_SOURCE_DIR}/Persistence/GUI_readline.cpp
	${CMAKE_SOURCE_DIR}/Persistence/serialize.cpp
	${CMAKE_SOURCE_DIR}/Persistence/GStringUtil.cpp
	${CMAKE_SOURCE_DIR}/Persistence/GUI_impl_readline.cpp
)
set(PROJ_INCLUDE
	${CMAKE_SOURCE_DIR}/GCVD/Addedutils.h
	${CMAKE_SOURCE_DIR}/GLWindow2.h
	${CMAKE_SOURCE_DIR}/GLWindowMenu.h
	${CMAKE_SOURCE_DIR}/VideoSource.h
	${CMAKE_SOURCE_DIR}/CalibImage.h
	${CMAKE_SOURCE_DIR}/CalibCornerPatch.h
	${CMAKE_SOURCE_DIR}/ATANCamera.h
	${CMAKE_SOURCE_DIR}/CameraCalibrator.h
	
	${CMAKE_SOURCE_DIR}/FAST/prototypes.h
	${CMAKE_SOURCE_DIR}/FAST/nonmax_suppression.h
	${CMAKE_SOURCE_DIR}/FAST/fast_corner.h
	
	${CMAKE_SOURCE_DIR}/GCVD/image_interpolate.h
	${CMAKE_SOURCE_DIR}/GCVD/Operators.h
	
	${CMAKE_SOURCE_DIR}/GCVD/SO3.h
	
	${CMAKE_SOURCE_DIR}/GCVD/SE3.h
	${CMAKE_SOURCE_DIR}/GCVD/GLWindow.h
	${CMAKE_SOURCE_DIR}/GCVD/GLFont.h
	${CMAKE_SOURCE_DIR}/GCVD/GLHelpers.h
	${CMAKE_SOURCE_DIR}/Persistence/default.h
	${CMAKE_SOURCE_DIR}/Persistence/serialize.h
	${CMAKE_SOURCE_DIR}/Persistence/type_name.h
	${CMAKE_SOURCE_DIR}/Persistence/PVars.h
	${CMAKE_SOURCE_DIR}/Persistence/GStringUtil.h
	${CMAKE_SOURCE_DIR}/Persistence/instances.h
	${CMAKE_SOURCE_DIR}/Persistence/GUI.h
	${CMAKE_SOURCE_DIR}/Persistence/GUI_impl.h
	${CMAKE_SOURCE_DIR}/Persistence/GUI_readline.h
	
     )
	
############## GUITEST SOURCE FILES DONE #########################

# The notorious OpenGL linker flags
SET( GL_LINKER_FLAGS "-lglut -lGLU -lGL")
# More linker flags for the pthread (had to look this one up and just copied it)
SET( PTHREAD_PROBLEM_LINKER_FLAGS "-lpthread -lm")
# And another linker flag for the use of readline.h
SET( GNU_READLINE_LINKER_FLAG "-lreadline")

# declaring external library include directories
include_directories(${EXT_INCLUDE_DIRS})

	       
add_executable(${PROJ_NAME}
	${PROJ_SOURCE}
	${PROJ_INCLUDE}
	
        )
        
set_property(TARGET ${PROJ_NAME} APPEND_STRING PROPERTY COMPILE_FLAGS "-D_LINUX -Wall -std=c++14 -march=native ")        

target_link_libraries(${PROJ_NAME}
		      ${EXT_LIBS}
		      ${GL_LINKER_FLAGS}
		      ${PTHREAD_PROBLEM_LINKER_FLAGS}
		      ${GNU_READLINE_LINKER_FLAG}
		      )


#install(TARGETS ${PROJ_NAME} RUNTIME DESTINATION ${CMAKE_SOURCE_DIR})

