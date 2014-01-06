#include "TargetDetector.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <TargetDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
TargetDetector::TargetDetector(Ptr<FeatureDetector> featureDetector, Ptr<DescriptorExtractor> descriptorExtractor, Ptr<DescriptorMatcher> descriptorMatcher, Scalar contourColor) :
	_featureDetector(featureDetector), _descriptorExtractor(descriptorExtractor), _descriptorMatcher(descriptorMatcher), _contourColor(contourColor) {}

TargetDetector::~TargetDetector() {}


bool TargetDetector::setupTargetRecognition(const Mat& targetImage, const Mat& targetROIs, size_t targetTag) {
	_targetImage = targetImage;	
	_targetTag = targetTag;

	// detect target keypoints
	_featureDetector->detect(_targetImage, _targetKeypoints, targetROIs);
	if (_targetKeypoints.size() < 4) { return false; }

	// compute descriptors
	_descriptorExtractor->compute(_targetImage, _targetKeypoints, _targetDescriptors);
	
	// train matcher to speedup recognition in case flann is used
	/*_descriptorMatcher->add(_targetDescriptors);
	_descriptorMatcher->train();*/

	// associate key points to ROIs
	return setupTargetROIs(_targetKeypoints, targetROIs);
}


bool TargetDetector::setupTargetROIs(const vector<KeyPoint>& targetKeypoints, const Mat& targetROIs) {
	if (targetKeypoints.empty()) { return false; }
	
	_targetKeypointsAssociatedROIsIndexes.resize(targetKeypoints.size());
		
	vector< vector<Point> > targetROIsContours;
	vector<Vec4i> hierarchy;
	cv::findContours(targetROIs.clone(), targetROIsContours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	if (targetROIsContours.empty()) { return false; }

	int targetKeypointsSize = targetKeypoints.size();
	#pragma omp parallel for
	for (int targetKeypointsIndex = 0; targetKeypointsIndex < targetKeypointsSize; ++targetKeypointsIndex) {
		for (size_t contourIndex = 0; contourIndex < targetROIsContours.size(); ++contourIndex) {
			// point inside contour or in the border
			Point2f point = targetKeypoints[targetKeypointsIndex].pt;
			if (cv::pointPolygonTest(targetROIsContours[contourIndex], point, false) >= 0) {
				_targetKeypointsAssociatedROIsIndexes[targetKeypointsIndex] = contourIndex;
				break;
			}
		}
	}
	
	_numberOfKeypointInsideContours.clear();
	_numberOfKeypointInsideContours.resize(targetROIsContours.size(), 0);

	for (size_t i = 0; i < _targetKeypointsAssociatedROIsIndexes.size(); ++i) {
		size_t contourIndex = _targetKeypointsAssociatedROIsIndexes[i];
		++_numberOfKeypointInsideContours[contourIndex];
	}

	return true;
}


Ptr<DetectorResult> TargetDetector::analyzeImage(const vector<KeyPoint>& keypointsQueryImage, const Mat& descriptorsQueryImage, float reprojectionThreshold) {	
	vector<DMatch> matches;
	_descriptorMatcher->match(descriptorsQueryImage, _targetDescriptors, matches);

	Mat homographyOut;
	vector<DMatch> inliersOut;
	vector<unsigned char> inliersMaskOut;
	ImageUtils::refineMatchesWithHomography(keypointsQueryImage, _targetKeypoints, matches, homographyOut, inliersOut, inliersMaskOut);
	vector<Point2f> contour;
	float bestROIMatch = (float)inliersOut.size() / (float)matches.size(); // TODO bestROIMatch
	
	return new DetectorResult(_targetTag, contour, _contourColor, bestROIMatch, _targetImage, _targetKeypoints, keypointsQueryImage, matches, inliersOut, inliersMaskOut, homographyOut);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </TargetDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
