#include "ImagePreprocessor.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <ImagePreprocessor> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImagePreprocessor::ImagePreprocessor(int claehClipLimit, int claehTileXSize, int claehTileYSize,
	int bilateralFilterDistance, int bilateralFilterSigmaColor, int bilateralFilterSigmaSpace,
	int contrastMultipliedBy10, int brightnessMultipliedBy10) :
		_claehClipLimit(claehClipLimit), _claehTileXSize(claehTileXSize), _claehTileYSize(claehTileYSize),
		_bilateralFilterDistance(bilateralFilterDistance), _bilateralFilterSigmaColor(bilateralFilterSigmaColor), _bilateralFilterSigmaSpace(bilateralFilterSigmaSpace),
		_contrast(contrastMultipliedBy10), _brightness(brightnessMultipliedBy10)
{}

ImagePreprocessor::~ImagePreprocessor() {}


bool ImagePreprocessor::loadAndPreprocessImage(const string& filename, Mat& imageLoadedOut, int loadFlags, bool useCVHighGUI) {
	if (filename != "") {
		try {
			imageLoadedOut = imread(filename, loadFlags);
			if (!imageLoadedOut.data) { return false; }
			preprocessImage(imageLoadedOut, useCVHighGUI);
			return true;
		} catch (...) {
			return false;
		}		
	}

	return false;
}


void ImagePreprocessor::preprocessImage(Mat& image, bool useCVHighGUI) {
	// remove noise with bilateral filter
	cv::bilateralFilter(image.clone(), image, _bilateralFilterDistance, _bilateralFilterSigmaColor, _bilateralFilterSigmaSpace);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_BILATERAL_FILTER, image);
	}

	// histogram equalization to improve color segmentation
	//histogramEqualization(image.clone(), false, useCVHighGUI);
	histogramEqualization(image, true, useCVHighGUI);

	// increase contrast and brightness
	image.convertTo(image, -1, (double)_contrast / 10.0, (double)_brightness / 10.0);

	cv::bilateralFilter(image.clone(), image, _bilateralFilterDistance, _bilateralFilterSigmaColor, _bilateralFilterSigmaSpace);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_CONTRAST_AND_BRIGHTNESS, image);
	}
}


void ImagePreprocessor::histogramEqualization(Mat& image, bool useCLAHE, bool useCVHighGUI) {
	vector<Mat> channels;
	if (image.channels() > 1) {
		cvtColor(image, image, CV_BGR2YCrCb);		
		cv::split(image, channels);
	}

	if (useCLAHE) {
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE((_claehClipLimit < 1 ? 1 : _claehClipLimit), Size((_claehTileXSize < 1 ? 1 : _claehTileXSize), (_claehTileYSize < 1 ? 1 : _claehTileYSize)));
		if (image.channels() > 1) {
			clahe->apply(channels[0], channels[0]);
		} else {
			clahe->apply(image, image);
		}
	} else {
		cv::equalizeHist(channels[0], channels[0]);
	}
	
	if (image.channels() > 1) {
		cv::merge(channels, image);
		cvtColor(image, image, CV_YCrCb2BGR);
	}

	if (useCVHighGUI) {
		if (useCLAHE) {
			imshow(WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE, image);
		}
		else {
			imshow(WINDOW_NAME_HISTOGRAM_EQUALIZATION, image);
		}
	}
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> </ImagePreprocessor> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
