#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iostream>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

// project includes
#include "ImagePreprocessor.h"
#include "ImageDetector.h"
#include "../GUI/GUIUtils.h"

// namespace specific imports to avoid namespace pollution
using std::string;
using std::stringstream;
using std::vector;
using std::map;
using std::pair;
using std::cout;

using cv::Mat;
using cv::Rect;
using cv::RotatedRect;
using cv::Ptr;
using cv::Scalar;
using cv::Vec3f;
using cv::Point;
using cv::Point2f;
using cv::Size;
using cv::VideoCapture;
using cv::imread;
using cv::waitKey;
using cv::imshow;
using cv::namedWindow;
using cv::moveWindow;
using cv::resizeWindow;
using cv::circle;
using cv::ellipse;
using cv::rectangle;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

/// Image analysis class that detects speed limits signs and recognizes the speed limit number
class ImageAnalysis {
	public:
		
		/// Constructor with initialization of parameters with default value		 		 
		ImageAnalysis(Ptr<ImagePreprocessor> imagePreprocessor, Ptr<ImageDetector> imageDetector);
		
		/// ImageAnalysis destructor that performs cleanup of OpenCV HighGUI windows (in case they are used)		 
		virtual ~ImageAnalysis();			


		/*!
		 * \brief Processes the image from the specified path
		 * \param filename Image name
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return true if image was successfully processed
		 */
		bool processImage(string filename, bool useCVHighGUI = true);


		/*!
		 * \brief Processes the image already loaded
		 * \param image Image loaded and ready to be processed
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return true if image was successfully processed
		 */
		bool processImage(Mat& image, bool useCVHighGUI = true);
						

		/*!
		 * \brief Processes the image to reflect any internal parameter change		 
		 * \return True if processing finished successfully
		 */
		bool updateImage();
		

		/*!
		 * \brief Processes a video from a file, analyzing the presence of speed limit signs
		 * \param path Full path to video
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return True if processing finished successfully
		 */
		bool processVideo(string path, bool useCVHighGUI = true);

		/*!
		 * \brief Processes a video from a camera, analyzing the presence of speed limit signs
		 * \param cameraDeviceNumber Camera device number
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return True if processing finished successfully
		 */
		bool processVideo(int cameraDeviceNumber, bool useCVHighGUI = true);


		/*!
		 * \brief Processes a video from a VideoCapture source, analyzing the presence of speed limit signs
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return True if processing finished successfully
		 */
		bool processVideo(VideoCapture videoCapture, bool useCVHighGUI = true);

		
		/// brief Setups the HighGUI window were the original image is going to be drawn		 		 
		void setupMainWindow();


		/*!
		 * \brief Setups the windows were the results will be presented
		 * \param optionsOneWindow Flag to indicate to group the track bars in one window		 
		 */
		void setupResultsWindows(bool optionsOneWindow = false);			


		// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
		int getScreenWidth() const { return _screenWidth; }
		void setScreenWidth(int val) { _screenWidth = val; }

		int getScreenHeight() const { return _screenHeight; }
		void setScreenHeight(int val) { _screenHeight = val; }

		bool getOptionsOneWindow() const { return _optionsOneWindow; }
		void setOptionsOneWindow(bool val) { _optionsOneWindow = val; }
		// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

	private:		
		Mat _originalImage;
		Mat _preprocessedImage;
		Mat _processedImage;
		bool _useCVHiGUI;
		bool _windowsInitialized;
		bool _optionsOneWindow;
		
		int _frameRate;
		int _screenWidth;		
		int _screenHeight;		

		Ptr<ImagePreprocessor> _imagePreprocessor;
		Ptr<ImageDetector> _imageDetector;

		string _filename;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
