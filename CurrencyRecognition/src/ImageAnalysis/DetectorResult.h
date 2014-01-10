#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>

// project includes
#include "../Configs.h"


// namespace specific imports to avoid namespace pollution
using std::vector;

using cv::Mat;
using cv::Point;
using cv::Point2f;
using cv::Scalar;
using cv::DMatch;
using cv::KeyPoint;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <DetectorResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class DetectorResult {
	public:
		DetectorResult();
		DetectorResult(size_t targetValue, const vector<Point>& targetContour, const Scalar& contourColor, float bestROIMatch,
			const Mat& referenceImage, const vector<KeyPoint>& referenceImageKeypoints, const vector<KeyPoint>& keypointsQueryImage,
			const vector<DMatch>& matches, const vector<DMatch>& inliers, const vector<unsigned char>& inliersMatchesMask, const Mat& homography);
		
		virtual ~DetectorResult();		


		// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
		size_t& getTargetValue() { return _targetValue; }		
		vector<Point>& getTargetContour();
		Scalar& getContourColor() { return _contourColor; }		
		float& getBestROIMatch() { return _bestROIMatch; }
		Mat& getReferenceImage() { return _referenceImage; }
		vector<KeyPoint>& getKeypointsQueryImage() { return _keypointsQueryImage; }
		vector<DMatch>& getMatches() { return _matches; }
		vector<DMatch>& getInliers() { return _inliers; }
		vector<KeyPoint>& getInliersKeypoints();
		Mat getInliersMatches(Mat& queryImage);
		vector<unsigned char>& getInliersMatchesMask() { return _inliersMatchesMask; }
		Mat& getHomography() { return _homography; }		
		// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

	protected:
		size_t _targetValue;
		vector<Point> _targetContour;
		Scalar _contourColor;
		float _bestROIMatch;

		Mat _referenceImage;
		vector<KeyPoint> _referenceImageKeypoints;
		vector<KeyPoint> _keypointsQueryImage;
		vector<DMatch> _matches;
		vector<DMatch> _inliers;
		vector<KeyPoint> _inliersKeyPoints;
		vector<unsigned char> _inliersMatchesMask;		

		Mat _homography;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </DetectorResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
