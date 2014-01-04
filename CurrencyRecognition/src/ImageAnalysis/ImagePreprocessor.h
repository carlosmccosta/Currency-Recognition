#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <vector>
#include <string>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

// project includes
#include "../Configs.h"

// namespace specific imports to avoid namespace pollution
using std::vector;
using std::string;

using cv::Mat;
using cv::Size;
using cv::imshow;
using cv::imread;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <ImagePreprocessor> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class ImagePreprocessor {
	public:
		ImagePreprocessor(int claehClipLimit = 2, int claehTileXSize = 4, int claehTileYSize = 4,
			int bilateralFilterDistance = 8, int bilateralFilterSigmaColor = 16, int bilateralFilterSigmaSpace = 12,
			int contrastMultipliedBy10 = 9, int brightnessMultipliedBy10 = 24);
		virtual ~ImagePreprocessor();
	

		bool loadAndPreprocessImage(const string& filename, Mat& imageLoadedOut, int loadFlags = CV_LOAD_IMAGE_COLOR, bool useCVHighGUI = false);

		/*!
		* \brief Preprocesses the image by applying bilateral filtering, histogram equalization, contrast and brightness correction and bilateral filtering again
		* \param image Image to be preprocessed
		* \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		*/
		void preprocessImage(Mat& image, bool useCVHighGUI = true);


		/*!
		* \brief Applies histogram equalization to the specified image
		* \param image Image to equalize
		* \param useCLAHE If true, uses the contrast limited adaptive histogram equalization (CLAHE)
		* \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		* \return
		*/
		void histogramEqualization(Mat& image, bool useCLAHE = true, bool useCVHighGUI = true);


		// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
		int getClaehClipLimit() const { return _claehClipLimit; }
		int* getClaehClipLimitPtr() { return &_claehClipLimit; }
		void setClaehClipLimit(int val) { _claehClipLimit = val; }
		int getClaehTileXSize() const { return _claehTileXSize; }
		int* getClaehTileXSizePtr() { return &_claehTileXSize; }
		void ClaehTileXSize(int val) { _claehTileXSize = val; }
		int getClaehTileYSize() const { return _claehTileYSize; }
		int* getClaehTileYSizePtr() { return &_claehTileYSize; }
		void setClaehTileYSize(int val) { _claehTileYSize = val; }

		int getBilateralFilterDistance() const { return _bilateralFilterDistance; }
		int* getBilateralFilterDistancePtr() { return &_bilateralFilterDistance; }
		void setBilateralFilterDistance(int val) { _bilateralFilterDistance = val; }
		int getBilateralFilterSigmaColor() const { return _bilateralFilterSigmaColor; }
		int* getBilateralFilterSigmaColorPtr() { return &_bilateralFilterSigmaColor; }
		void setBilateralFilterSigmaColor(int val) { _bilateralFilterSigmaColor = val; }
		int getBilateralFilterSigmaSpace() const { return _bilateralFilterSigmaSpace; }
		int* getBilateralFilterSigmaSpacePtr() { return &_bilateralFilterSigmaSpace; }
		void setBilateralFilterSigmaSpace(int val) { _bilateralFilterSigmaSpace = val; }

		int getContrast() const { return _contrast; }
		int* getContrastPtr() { return &_contrast; }
		void setContrast(int val) { _contrast = val; }
		int getBrightness() const { return _brightness; }
		int* getBrightnessPtr() { return &_brightness; }
		void setBrightness(int val) { _brightness = val; }
		// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------


	private:
		int _claehClipLimit;
		int _claehTileXSize;
		int _claehTileYSize;		

		int _bilateralFilterDistance;		
		int _bilateralFilterSigmaColor;		
		int _bilateralFilterSigmaSpace;
		
		int _contrast;		
		int _brightness;		
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> </ImagePreprocessor> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
