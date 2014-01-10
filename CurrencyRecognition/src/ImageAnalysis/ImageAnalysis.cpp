#include "ImageAnalysis.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageAnalysis::ImageAnalysis(Ptr<ImagePreprocessor> imagePreprocessor, Ptr<ImageDetector> imageDetector) :
	_useCVHiGUI(true), _windowsInitialized(false), _optionsOneWindow(false),
	_frameRate(30), _screenWidth(1920), _screenHeight(1080),
	_imagePreprocessor(imagePreprocessor), _imageDetector(imageDetector) {};


ImageAnalysis::~ImageAnalysis() {
	if (_useCVHiGUI) {
		cv::destroyAllWindows();
	}
}


bool ImageAnalysis::processImage(string filename, bool useCVHighGUI) {				
	Mat imageToProcess;
	bool loadSuccessful = true;
	if (filename != "") {
		try {
			imageToProcess = imread(TEST_IMGAGES_DIRECTORY + filename, CV_LOAD_IMAGE_GRAYSCALE);
		} catch (...) {
			loadSuccessful = false;
		}			

		if (!imageToProcess.data) {
			loadSuccessful = false;
		}
	} else {		
		loadSuccessful = false;
	}

	if (!loadSuccessful) {
		if (useCVHighGUI) {
			cv::destroyAllWindows();
		}

		return false;
	}

	_useCVHiGUI = useCVHighGUI;
	_windowsInitialized = false;

	_filename = filename;
	bool status = processImage(imageToProcess, useCVHighGUI);	
	_filename = "";

	while(waitKey(10) != ESC_KEYCODE) {}
	
	if (useCVHighGUI) {
		cv::destroyAllWindows();
	}

	return status;
}


bool ImageAnalysis::processImage(Mat& image, bool useCVHighGUI) {
	_originalImage = image.clone();	
	_useCVHiGUI = useCVHighGUI;
	
	if (useCVHighGUI) {		
		if (!_windowsInitialized) {
			setupMainWindow();
			setupResultsWindows(_optionsOneWindow);
			_windowsInitialized = true;
		}

		imshow(WINDOW_NAME_MAIN, _originalImage);
	}

	_preprocessedImage = image.clone();
	_imagePreprocessor->preprocessImage(_preprocessedImage, useCVHighGUI);
	_processedImage = _preprocessedImage.clone();

	_imageDetector->detectTargetsAndOutputResults(_processedImage, _filename, true);

	imshow(WINDOW_NAME_TARGET_DETECTION, _processedImage);	

	return true;
}


bool ImageAnalysis::updateImage() {
	return processImage(_originalImage.clone(), _useCVHiGUI);
}



// -------------------------------------------------------------------------------------  <Video processing>  -----------------------------------------------------------------------------------------
bool ImageAnalysis::processVideo(string path, bool useCVHighGUI) {	
	VideoCapture videoCapture;
	
	try {
		videoCapture = VideoCapture(path);
	} catch (...) {
		return false;
	}

	return processVideo(videoCapture, useCVHighGUI);
}


bool ImageAnalysis::processVideo(int cameraDeviceNumber, bool useCVHighGUI) {	
	VideoCapture videoCapture;

	try {
		videoCapture = VideoCapture(cameraDeviceNumber);
	} catch (...) {
		return false;
	}

	return processVideo(videoCapture, useCVHighGUI);
}


bool ImageAnalysis::processVideo(VideoCapture videoCapture, bool useCVHighGUI) {		
	if (!videoCapture.isOpened()) {
		return false;
	}

	_useCVHiGUI = useCVHighGUI;
	_windowsInitialized = false;

	int millisecPollingInterval = 1000 / _frameRate;
	if (millisecPollingInterval < 10)
		millisecPollingInterval = 10;
	
	Mat currentFrame;	
	while (videoCapture.read(currentFrame)) {
		try {
			processImage(currentFrame, useCVHighGUI);
		} catch(...) {}
		
		if (waitKey(millisecPollingInterval) == ESC_KEYCODE) {
			break;
		}
	}

	if (useCVHighGUI) {
		cv::destroyAllWindows();
	}

	return true;
}
// -------------------------------------------------------------------------------------  </Video processing>  ----------------------------------------------------------------------------------------



// --------------------------------------------------------------------------------------  <OpenCV HighGUI>  ------------------------------------------------------------------------------------------
void updateImageAnalysis(int position, void* userData) {		
	ImageAnalysis* imgAnalysis = ((ImageAnalysis*)userData);
	imgAnalysis->updateImage();
}


void ImageAnalysis::setupMainWindow() {
	GUIUtils::addHighGUIWindow(0, 0, WINDOW_NAME_MAIN, _originalImage.size().width, _originalImage.size().height, _screenWidth, _screenHeight);
}


void ImageAnalysis::setupResultsWindows(bool optionsOneWindow) {
	GUIUtils::addHighGUIWindow(1, 0, WINDOW_NAME_BILATERAL_FILTER, _originalImage.size().width, _originalImage.size().height, _screenWidth, _screenHeight);
	//GUIUtils::addHighGUIWindow(2, 0, WINDOW_NAME_HISTOGRAM_EQUALIZATION, _originalImage.size().width, _originalImage.size().height, _screenWidth, _screenHeight);
	GUIUtils::addHighGUIWindow(2, 0, WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE, _originalImage.size().width, _originalImage.size().height, _screenWidth, _screenHeight);
	GUIUtils::addHighGUIWindow(0, 1, WINDOW_NAME_CONTRAST_AND_BRIGHTNESS, _originalImage.size().width, _originalImage.size().height, _screenWidth, _screenHeight);	
	GUIUtils::addHighGUIWindow(1, 1, WINDOW_NAME_TARGET_DETECTION, _originalImage.size().width, _originalImage.size().height, _screenWidth, _screenHeight);
	
	if (optionsOneWindow) {		
		namedWindow(WINDOW_NAME_OPTIONS, CV_WINDOW_NORMAL);
		resizeWindow(WINDOW_NAME_OPTIONS, WINDOW_OPTIONS_WIDTH - WINDOW_FRAME_THICKNESS * 2, WINDOW_OPTIONS_HIGHT);
		moveWindow(WINDOW_NAME_OPTIONS, _screenWidth - WINDOW_OPTIONS_WIDTH, 0);
	} else {						
		GUIUtils::addHighGUITrackBarWindow(WINDOW_NAME_BILATERAL_FILTER_OPTIONS, 3, 0, 0, _screenWidth);		
		GUIUtils::addHighGUITrackBarWindow(WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS, 3, 3, 1, _screenWidth);
		GUIUtils::addHighGUITrackBarWindow(WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS, 2, 6, 2, _screenWidth);
	}	
	
	cv::createTrackbar(TRACK_BAR_NAME_BILATERAL_FILTER_DIST, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), _imagePreprocessor->getBilateralFilterDistancePtr(), 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BILATERAL_FILTER_COLOR_SIG, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), _imagePreprocessor->getBilateralFilterSigmaColorPtr(), 200, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BILATERAL_FILTER_SPACE_SIG, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), _imagePreprocessor->getBilateralFilterSigmaSpacePtr(), 200, updateImageAnalysis, (void*)this);
	
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_CLIP, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), _imagePreprocessor->getClaehClipLimitPtr(), 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_TILE_X, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), _imagePreprocessor->getClaehTileXSizePtr(), 20, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_TILE_Y, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), _imagePreprocessor->getClaehTileYSizePtr(), 20, updateImageAnalysis, (void*)this);

	cv::createTrackbar(TRACK_BAR_NAME_CONTRAST, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS), _imagePreprocessor->getContrastPtr(), 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BRIGHTNESS, (optionsOneWindow ? WINDOW_NAME_OPTIONS : WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS), _imagePreprocessor->getBrightnessPtr(), 1000, updateImageAnalysis, (void*)this);
}
// --------------------------------------------------------------------------------------  </OpenCV HighGUI>  -----------------------------------------------------------------------------------------

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<