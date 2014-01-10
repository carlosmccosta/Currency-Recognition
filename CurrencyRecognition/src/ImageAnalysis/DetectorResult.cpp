#include "DetectorResult.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <DetectorResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
DetectorResult::DetectorResult() : _bestROIMatch(0) {}

DetectorResult::DetectorResult(size_t targetValue, const vector<Point>& targetContour, const Scalar& contourColor, float bestROIMatch,
	const Mat& referenceImage, const vector<KeyPoint>& referenceImageKeypoints, const vector<KeyPoint>& keypointsQueryImage,
	const vector<DMatch>& matches, const vector<DMatch>& inliers, const vector<unsigned char>& inliersMatchesMask, const Mat& homography) :
	
	_targetValue(targetValue), _targetContour(targetContour), _contourColor(contourColor), _bestROIMatch(bestROIMatch),
	_referenceImage(referenceImage), _referenceImageKeypoints(referenceImageKeypoints), _keypointsQueryImage(keypointsQueryImage),
	_matches(matches), _inliers(inliers), _inliersMatchesMask(inliersMatchesMask), _homography(homography) {}


DetectorResult::~DetectorResult() {}


vector<Point>& DetectorResult::getTargetContour() {
	if (_targetContour.empty()) {
		vector<Point2f> corners;
		corners.push_back(Point2f(0.0f, 0.0f));
		corners.push_back(Point2f((float)_referenceImage.cols, 0.0f));
		corners.push_back(Point2f((float)_referenceImage.cols, (float)_referenceImage.rows));
		corners.push_back(Point2f(0.0f, (float)_referenceImage.rows));

		vector<Point2f> transformedCorners;
		cv::perspectiveTransform(corners, transformedCorners, _homography);

		for (size_t i = 0; i < transformedCorners.size(); ++i) {
			_targetContour.push_back(Point((int)transformedCorners[i].x, (int)transformedCorners[i].y));
		}
	}	

	return _targetContour;
}


vector<KeyPoint>& DetectorResult::getInliersKeypoints() {
	if (_inliersKeyPoints.empty()) {
		for (size_t i = 0; i < _inliers.size(); ++i) {
			DMatch match = _inliers[i];

			if ((size_t)match.queryIdx < _keypointsQueryImage.size()) {
				_inliersKeyPoints.push_back(_keypointsQueryImage[match.queryIdx]);				
			}
		}
	}

	return _inliersKeyPoints;
}


Mat DetectorResult::getInliersMatches(Mat& queryImage) {	
	Mat inliersMatches;

	if (_inliers.empty()) {
		return queryImage;
	} else {
		cv::drawMatches(queryImage, _keypointsQueryImage, _referenceImage, _referenceImageKeypoints, _inliers, inliersMatches, TARGET_KEYPOINT_COLOR, NONTARGET_KEYPOINT_COLOR);
		return inliersMatches;
	}
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </DetectorResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
