#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#define INLIERS_MATCHES "inliersMatches"
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
#include <sstream>
#include <vector>
#include <fstream>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

// project includes
#include "../Configs.h"
#include "ImagePreprocessor.h"
#include "DetectorEvaluationResult.h"
#include "TargetDetector.h"
#include "DetectorResult.h"
#include "../libs/PerformanceTimer.h"
#include "../GUI/GUIUtils.h"

// namespace specific imports to avoid namespace pollution
using std::cout;
using std::endl;
using std::string;
using std::stringstream;
using std::vector;
using std::ifstream;
using std::ofstream;

using cv::Mat;
using cv::Ptr;
using cv::Rect;
using cv::FeatureDetector;
using cv::DescriptorExtractor;
using cv::DescriptorMatcher;
using cv::imwrite;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class ImageDetector {
	public:
		ImageDetector(Ptr<FeatureDetector> featureDetector, Ptr<DescriptorExtractor> descriptorExtractor, Ptr<DescriptorMatcher> descriptorMatcher, Ptr<ImagePreprocessor> imagePreprocessor,
			const string& configurationTags, const vector<string>& referenceImagesDirectories,
			bool useInliersGlobalMatch = true,
			const string& referenceImagesListPath = REFERENCE_IMGAGES_LIST, const string& testImagesListPath = TEST_IMGAGES_LIST);
		virtual ~ImageDetector();

		bool setupTargetDB(const string& referenceImagesListPaths, bool useInliersGlobalMatch = true);

		virtual Ptr< vector< Ptr<DetectorResult> > > detectTargets(Mat& image, float minimumMatchAllowed = 0.07, float minimumTargetAreaPercentage = 0.05,
			float maxDistanceRatio = 0.75f, float reprojectionThresholdPercentage = 0.01f, double confidence = 0.999, int maxIters = 5000, size_t minimumNumberInliers = 6);
		virtual vector<size_t> detectTargetsAndOutputResults(Mat& image, string imageFilenameWithoutExtension = "", bool useHighGUI = false);
		DetectorEvaluationResult evaluateDetector(const string& testImgsList, bool saveResults = true);

	protected:						
		Ptr<FeatureDetector> _featureDetector;
		Ptr<DescriptorExtractor> _descriptorExtractor;
		Ptr<DescriptorMatcher> _descriptorMatcher;

		Ptr<ImagePreprocessor> _imagePreprocessor;
		string _configurationTags;
		vector<string> _referenceImagesDirectories;
		string _referenceImagesListPath;
		string _testImagesListPath;

		vector<TargetDetector> _targetDetectors;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
