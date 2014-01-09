#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <iostream>
#include <string>
#include <sstream>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>

// project includes
#include "../Configs.h"
#include "ConsoleInput.h"
#include "../ImageAnalysis/ImageAnalysis.h"
#include "../ImageAnalysis/ImageDetector.h"


// namespace specific imports to avoid namespace pollution
using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::stringstream;

using cv::FeatureDetector;
using cv::DescriptorExtractor;
using cv::DescriptorMatcher;
using cv::BOWTrainer;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Commnd Line user Interface>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class CLI {
	public:
		CLI() : _imagePreprocessor(new ImagePreprocessor()) {}
		virtual ~CLI() {}

		void startInteractiveCLI();
		void showConsoleHeader();
		int getUserOption();
		void setupImageRecognition();

		int selectImagesDBLevelOfDetail();
		int selectInliersSelectionMethod();
		int selectFeatureDetector();
		int selectDescriptorExtractor();
		int selectDescriptorMatcher();		
		
		void showVersion();

	private:
		Ptr<ImagePreprocessor> _imagePreprocessor;		
		Ptr<ImageDetector> _imageDetector;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </Commnd Line user Interface>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
