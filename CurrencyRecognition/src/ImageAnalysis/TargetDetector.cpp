#include "TargetDetector.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <TargetDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
TargetDetector::TargetDetector(Ptr<FeatureDetector> featureDetector, Ptr<DescriptorExtractor> descriptorExtractor, Ptr<DescriptorMatcher> descriptorMatcher,
	size_t targetTag, const Scalar& contourColor, bool useInliersGlobalMatch) :
	_featureDetector(featureDetector), _descriptorExtractor(descriptorExtractor), _descriptorMatcher(descriptorMatcher/*->clone(true)*/),
	_targetTag(targetTag), _contourColor(contourColor), _useInliersGlobalMatch(useInliersGlobalMatch),
	_currentLODIndex(0) {}

TargetDetector::~TargetDetector() {}


bool TargetDetector::setupTargetRecognition(const Mat& targetImage, const Mat& targetROIs) {
	_targetsImage.push_back(targetImage);	
	_targetsKeypoints.push_back(vector<KeyPoint>());
	_targetsDescriptors.push_back(Mat());
	_currentLODIndex = _targetsKeypoints.size() - 1;

	// detect target keypoints
	_featureDetector->detect(_targetsImage[_currentLODIndex], _targetsKeypoints[_currentLODIndex], targetROIs);
	if (_targetsKeypoints[_currentLODIndex].size() < 4) { return false; }

	// compute descriptors
	_descriptorExtractor->compute(_targetsImage[_currentLODIndex], _targetsKeypoints[_currentLODIndex], _targetsDescriptors[_currentLODIndex]);
	if (_targetsDescriptors[_currentLODIndex].rows < 4) { return false; }


	// train matcher to speedup recognition in case flann is used
	/*_descriptorMatcher->add(_targetDescriptors);
	_descriptorMatcher->train();*/

	// associate key points to ROIs
	if (!_useInliersGlobalMatch) {
		return setupTargetROIs(_targetsKeypoints[_currentLODIndex], targetROIs);
	} else {
		return true;
	}
}


bool TargetDetector::setupTargetROIs(const vector<KeyPoint>& targetKeypoints, const Mat& targetROIs) {
	_targetKeypointsAssociatedROIsIndexes.push_back(vector<size_t>());
	_numberOfKeypointInsideContours.push_back(vector<size_t>());

	if (targetKeypoints.empty()) { return false; }
	
	_targetKeypointsAssociatedROIsIndexes[_currentLODIndex].resize(targetKeypoints.size());
		
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
				_targetKeypointsAssociatedROIsIndexes[_currentLODIndex][targetKeypointsIndex] = contourIndex;
				break;
			}
		}
	}
	
	_numberOfKeypointInsideContours[_currentLODIndex].clear();
	_numberOfKeypointInsideContours[_currentLODIndex].resize(targetROIsContours.size(), 0);

	for (size_t i = 0; i < _targetKeypointsAssociatedROIsIndexes[_currentLODIndex].size(); ++i) {
		size_t contourIndex = _targetKeypointsAssociatedROIsIndexes[_currentLODIndex][i];
		++_numberOfKeypointInsideContours[_currentLODIndex][contourIndex];
	}

	return true;
}


void TargetDetector::updateCurrentLODIndex(const Mat& imageToAnalyze, float targetResolutionSelectionSplitOffset) {
	int halfImageResolution = imageToAnalyze.cols / 2;

	size_t newLODIndex = 0;
	for (size_t i = 1; i < _targetsImage.size(); ++i) {
		int previousLODWidthResolution = _targetsImage[i - 1].cols;
		int currentLODWidthResolution = _targetsImage[i].cols;

		if (halfImageResolution > currentLODWidthResolution) {
			newLODIndex = i; // use bigger resolution
		} else if (halfImageResolution < previousLODWidthResolution) {
			newLODIndex = i - 1; // use lower resolution
			break;
		} else {
			int splittingPointResolutions = (int)((currentLODWidthResolution - previousLODWidthResolution) * targetResolutionSelectionSplitOffset);
			int imageOffsetResolution = currentLODWidthResolution - halfImageResolution;

			if (imageOffsetResolution < splittingPointResolutions) {
				newLODIndex = i - 1; // use lower resolution
				break;
			} else {
				newLODIndex = i; // use bigger resolution
				break;
			}
		}
	}

	_currentLODIndex = newLODIndex;
}


Ptr<DetectorResult> TargetDetector::analyzeImage(const vector<KeyPoint>& keypointsQueryImage, const Mat& descriptorsQueryImage,
	float maxDistanceRatio, float reprojectionThreshold, double confidence, int maxIters, size_t minimumNumberInliers) {
	vector<DMatch> matches;
	ImageUtils::matchDescriptorsWithRatioTest(_descriptorMatcher, descriptorsQueryImage, _targetsDescriptors[_currentLODIndex], matches, maxDistanceRatio);
	//_descriptorMatcher->match(descriptorsQueryImage, _targetDescriptors, matches);
	//_descriptorMatcher->match(descriptorsQueryImage, matches); // flann speedup

	if (matches.size() < minimumNumberInliers) {
		return new DetectorResult();
	}

	Mat homography;
	vector<DMatch> inliers;
	vector<unsigned char> inliersMaskOut;
	ImageUtils::refineMatchesWithHomography(keypointsQueryImage, _targetsKeypoints[_currentLODIndex], matches, homography, inliers, inliersMaskOut, reprojectionThreshold, confidence, maxIters, minimumNumberInliers);
	
	if (inliers.size() < minimumNumberInliers) {
		return new DetectorResult();
	}

	float bestROIMatch = 0;
	if (_useInliersGlobalMatch) {
		bestROIMatch = (float)inliers.size() / (float)matches.size();
	} else {
		bestROIMatch = computeBestROIMatch(inliers, minimumNumberInliers);
	}	
	
	return new DetectorResult(_targetTag, vector<Point>(), _contourColor, bestROIMatch, _targetsImage[_currentLODIndex], _targetsKeypoints[_currentLODIndex], keypointsQueryImage, matches, inliers, inliersMaskOut, homography);
}


float TargetDetector::computeBestROIMatch(const vector<DMatch>& inliers, size_t minimumNumberInliers) {
	vector<size_t> roisInliersCounts(_numberOfKeypointInsideContours[_currentLODIndex].size(), 0);

	for (size_t i = 0; i < inliers.size(); ++i) {
		size_t roiIndex = _targetKeypointsAssociatedROIsIndexes[_currentLODIndex][inliers[i].trainIdx];
		++roisInliersCounts[roiIndex];
	}

	float bestROIMatch = 0;
	for (size_t i = 0; i < roisInliersCounts.size(); ++i) {
		size_t roiTotalCount = _numberOfKeypointInsideContours[_currentLODIndex][i];
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
