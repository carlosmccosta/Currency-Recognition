#pragma once

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


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
		TargetDetector(Ptr<FeatureDetector> featureDetector, Ptr<DescriptorExtractor> descriptorExtractor, Ptr<DescriptorMatcher> descriptorMatcher, Scalar contourColor = Scalar(1,1,1));
		virtual ~TargetDetector();		

		bool setupTargetRecognition(const Mat& targetImage, const Mat& targetROIs, size_t targetTag);
		bool setupTargetROIs(const vector<KeyPoint>& targetKeypoints, const Mat& targetROIs);

		Ptr<DetectorResult> analyzeImage(const vector<KeyPoint>& keypointsQueryImage, const Mat& descriptorsQueryImage, size_t minimumNumberInliers = 6, float reprojectionThreshold = 3.0f);
		float computeBestROIMatch(const vector<DMatch>& inliers, size_t minimumNumberInliers = 6);

		// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
		size_t getTargetTag() const { return _targetTag; }
		void setTargetTag(size_t val) { _targetTag = val; }
		Mat getTargetImage() const { return _targetImage; }
		void setTargetImage(Mat val) { _targetImage = val; }		
		vector<KeyPoint>& getTargetKeypoints() { return _targetKeypoints; }
		void setTargetKeypoints(vector<KeyPoint> val) { _targetKeypoints = val; }
		// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

	protected:	
		Ptr<FeatureDetector> _featureDetector;
		Ptr<DescriptorExtractor> _descriptorExtractor;
		Ptr<DescriptorMatcher> _descriptorMatcher;
	
		size_t _targetTag;				
		Mat _targetImage;
		Scalar _contourColor;
				
		vector<KeyPoint> _targetKeypoints;		
		vector<size_t> _targetKeypointsAssociatedROIsIndexes;
		vector<size_t> _numberOfKeypointInsideContours;

		Mat _targetDescriptors;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </TargetDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
