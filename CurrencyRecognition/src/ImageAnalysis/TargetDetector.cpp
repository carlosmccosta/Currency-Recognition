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


Ptr<DetectorResult> TargetDetector::analyzeImage(const vector<KeyPoint>& keypointsQueryImage, const Mat& descriptorsQueryImage, size_t minimumNumberInliers, float reprojectionThreshold) {
	vector<DMatch> matches;
	ImageUtils::matchDescriptorsWithRatioTest(_descriptorMatcher, descriptorsQueryImage, _targetDescriptors, matches);
	//_descriptorMatcher->match(descriptorsQueryImage, _targetDescriptors, matches);
	//_descriptorMatcher->match(descriptorsQueryImage, matches); // flann speedup

	if (matches.size() < minimumNumberInliers) {
		return new DetectorResult();
	}

	Mat homography;
	vector<DMatch> inliers;
	vector<unsigned char> inliersMaskOut;
	ImageUtils::refineMatchesWithHomography(keypointsQueryImage, _targetKeypoints, matches, homography, inliers, inliersMaskOut, reprojectionThreshold);
	
	if (inliers.size() < minimumNumberInliers) {
		return new DetectorResult();
	}


	float bestROIMatch = (float)inliers.size() / (float)matches.size(); // global match
	//float bestROIMatch = computeBestROIMatch(inliers, minimumNumberInliers);
	
	return new DetectorResult(_targetTag, vector<Point2f>(), _contourColor, bestROIMatch, _targetImage, _targetKeypoints, keypointsQueryImage, matches, inliers, inliersMaskOut, homography);
}


float TargetDetector::computeBestROIMatch(const vector<DMatch>& inliers, size_t minimumNumberInliers) {
	vector<size_t> roisInliersCounts(_numberOfKeypointInsideContours.size(), 0);

	for (size_t i = 0; i < inliers.size(); ++i) {
		size_t roiIndex = _targetKeypointsAssociatedROIsIndexes[inliers[i].trainIdx];
		++roisInliersCounts[roiIndex];
	}

	float bestROIMatch = 0;
	for (size_t i = 0; i < roisInliersCounts.size(); ++i) {
		size_t roiTotalCount = _numberOfKeypointInsideContours[i];
		if (roiTotalCount != 0) {
			float roiMatch = (float)roisInliersCounts[i] / (float)roiTotalCount;
			if (roiMatch > bestROIMatch && roisInliersCounts[i] > minimumNumberInliers) {
				bestROIMatch = roiMatch;
			}
		}
	}

	return bestROIMatch;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </TargetDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
