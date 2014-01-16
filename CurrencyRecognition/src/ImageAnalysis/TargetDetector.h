#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <string>
#include <vector>
#include <unordered_map>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>


// project includes
#include "ImageUtils.h"
#include "DetectorResult.h"

// namespace specific imports to avoid namespace pollution
using std::string;
using std::vector;
using std::unordered_map;

using cv::Mat;
using cv::Ptr;
using cv::Rect;
using cv::FeatureDetector;
using cv::DescriptorExtractor;
using cv::DescriptorMatcher;
using cv::Scalar;
using cv::Point;
using cv::Point2f;
using cv::Vec4i;
using cv::KeyPoint;
using cv::DMatch;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <TargetDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class TargetDetector {
	public:
	TargetDetector(Ptr<FeatureDetector> featureDetector, Ptr<DescriptorExtractor> descriptorExtractor, Ptr<DescriptorMatcher> descriptorMatcher,
		size_t targetTag, const Scalar& contourColor = Scalar(1, 1, 1), bool useInliersGlobalMatch = true);
		virtual ~TargetDetector();		

		bool setupTargetRecognition(const Mat& targetImage, const Mat& targetROIs);
		bool setupTargetROIs(const vector<KeyPoint>& targetKeypoints, const Mat& targetROIs);

		void updateCurrentLODIndex(const Mat& imageToAnalyze, float targetResolutionSelectionSplitOffset = 0.7);
		Ptr<DetectorResult> analyzeImage(const vector<KeyPoint>& keypointsQueryImage, const Mat& descriptorsQueryImage,
			float maxDistanceRatio = 0.75f, float reprojectionThreshold = 3.0f, double confidence = 0.999, int maxIters = 5000, size_t minimumNumberInliers = 6);
		float computeBestROIMatch(const vector<DMatch>& inliers, size_t minimumNumberInliers = 6);

		// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
		size_t getTargetTag() const { return _targetTag; }
		void setTargetTag(size_t val) { _targetTag = val; }
		Mat& getTargetImage() { return _targetsImage[_currentLODIndex]; }
		void setTargetImage(Mat val) { _targetsImage[_currentLODIndex] = val; }
		vector<KeyPoint>& getTargetKeypoints() { return _targetsKeypoints[_currentLODIndex]; }
		void setTargetKeypoints(vector<KeyPoint> val) { _targetsKeypoints[_currentLODIndex] = val; }
		// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

	protected:	
		Ptr<FeatureDetector> _featureDetector;
		Ptr<DescriptorExtractor> _descriptorExtractor;
		Ptr<DescriptorMatcher> _descriptorMatcher;
	
		size_t _targetTag;		
		Scalar _contourColor;
				
		vector<Mat> _targetsImage;
		vector< vector<KeyPoint> > _targetsKeypoints;		
		vector< vector<size_t> > _targetKeypointsAssociatedROIsIndexes;
		vector< vector<size_t> > _numberOfKeypointInsideContours;
		vector<Mat> _targetsDescriptors;


		size_t _currentLODIndex;
		bool _useInliersGlobalMatch;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </TargetDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
