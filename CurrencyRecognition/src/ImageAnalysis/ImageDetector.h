#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#define DETECTION_MASK "detectionMask"
#define RESULTS_FILE "resultsAnalysis.txt"
#define RESULTS_FILE_HEADER ">>>>> Detector image results analysis <<<<<"
#define RESULTS_FILE_FOOTER ">>>>> Detector global results analysis <<<<<"
#define PRECISION_TOKEN "Precision"
#define RECALL_TOKEN "Recall"
#define ACCURACY_TOKEN "Accuracy"
#define GLOBAL_PRECISION_TOKEN "Global precision"
#define GLOBAL_RECALL_TOKEN "Global recall"
#define GLOBAL_ACCURACY_TOKEN "Global accuracy"
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// project includes
#include "ImagePreprocessor.h"
#include "DetectorEvaluationResult.h"
#include "../Configs.h"
#include "../libs/PerformanceTimer.h"

// namespace specific imports to avoid namespace pollution
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;

using cv::Mat;
using cv::Ptr;
using cv::Rect;
using cv::imwrite;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class ImageDetector {
	public:
		ImageDetector();
		virtual ~ImageDetector();

		virtual void detectTargets(Mat& image, vector<Rect>& targetsBoundingRectanglesOut, Mat& imageDetectionMasksOut, bool showTargetBoundingRectangles = true, bool showImageKeyPoints = true);
		DetectorEvaluationResult evaluateDetector(string testImgsList, bool saveResults = true);

	protected:
		string _configuration;
		ImagePreprocessor _imagePreprocessor;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
