#include "ImageUtils.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
void ImageUtils::loadImageMasks(string imagePath, vector<Mat>& masks) {
	size_t imageMaskNumber = 0;
	bool masksAvailable = true;

	while (masksAvailable) {
		stringstream imageMaskPath;
		imageMaskPath << imagePath << MASK_TOKEN << imageMaskNumber << MASK_EXTENSION;
		
		Mat mask = imread(imageMaskPath.str(), CV_LOAD_IMAGE_COLOR);
		if (mask.data) {
			masks.push_back(mask);
		} else {
			masksAvailable = false;
		}

		++imageMaskNumber;
	}
}


void ImageUtils::retriveTargetsMasks(string imagePath, vector<Mat>& masksOut, Scalar lowerRange, Scalar higherRange) {
	loadImageMasks(imagePath, masksOut);
	int masksSize = masksOut.size();

	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < masksSize; ++i) {
		/*Mat& before = masks[i];
		Mat after;
		cv::inRange(before, lowerRange, higherRange, after);
		masks[i] = after;*/

		cv::inRange(masksOut[i], lowerRange, higherRange, masksOut[i]);
	}
}


bool ImageUtils::mergeTargetMasks(vector<Mat>& masks, Mat& mergedMaskOut) {
	int masksSize = masks.size();
	if (masksSize == 0){
		return false;
	}

	int maxRows = 0;
	int maxCols = 0;
	for (int i = 0; i < masksSize; ++i) {
		int rowsMask = masks[i].rows;
		int colsMask = masks[i].cols;
		if (rowsMask > maxRows) {
			maxRows = rowsMask;
		}

		if (colsMask > maxCols) {
			maxCols = colsMask;
		}
	}	
	
	if (maxRows == 0 || maxCols == 0) {
		return false;
	}

	mergedMaskOut = Mat::zeros(maxRows, maxCols, CV_8UC1);
	for (int i = 0; i < masksSize; ++i) {
		if (masks[i].rows == maxRows && masks[i].cols == maxCols) {
			cv::bitwise_or(mergedMaskOut, masks[i], mergedMaskOut);
		}
	}	

	return true;
}


void ImageUtils::splitKeyPoints(string imagePath, const vector<KeyPoint>& keypoints, vector< vector <KeyPoint> >& keypointsTargetClass, vector<KeyPoint>& keypointsNonTargetClass) {
	keypointsTargetClass.clear();
	keypointsNonTargetClass.clear();

	vector<Mat> masks;
	loadImageMasks(imagePath, masks);
	int keyPointsSize = keypoints.size();	

	keypointsTargetClass.resize(masks.size());

	#pragma omp parallel for schedule(dynamic)
	for (int keyPointPosition = 0; keyPointPosition < keyPointsSize; ++keyPointPosition) {
		bool keyPointIsNotCar = true;

		for (size_t maskPosition = 0; maskPosition < masks.size(); ++maskPosition) {									
			Vec3b maskColorInKeyPointPosition = masks[maskPosition].at<Vec3b>(keypoints[keyPointPosition].pt);
						
			if (maskColorInKeyPointPosition[2] == 255) {
				#pragma omp critical
				keypointsTargetClass[maskPosition].push_back(keypoints[keyPointPosition]);
				
				keyPointIsNotCar = false;
				break;
			}			
		}
		
		if (keyPointIsNotCar) {
			#pragma omp critical
			keypointsNonTargetClass.push_back(keypoints[keyPointPosition]);
		}
	}
}


void ImageUtils::findMaskBoundingRectangles(Mat& mask, vector<Rect>& targetsBoundingRectanglesOut) {
	targetsBoundingRectanglesOut.clear();
	
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > contours_poly(contours.size());
	targetsBoundingRectanglesOut.resize(contours.size());	
	int contoursSize = contours.size();

	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < contoursSize; ++i) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		targetsBoundingRectanglesOut[i] = boundingRect(Mat(contours_poly[i]));
	}
}


bool ImageUtils::loadMatrix(string filename, string tag, Mat& matrixOut) {
	FileStorage fs;
	if (fs.open(filename, FileStorage::READ)) {		
		fs[tag] >> matrixOut;

		fs.release();
		return true;
	}

	return false;
}


bool ImageUtils::saveMatrix(string filename, string tag, const Mat& matrix) {
	FileStorage fs;
	if (fs.open(filename, FileStorage::WRITE)) {		
		fs << tag << matrix;

		fs.release();
		return true;
	}

	return false;
}



bool ImageUtils::refineMatchesWithHomography(const vector<KeyPoint>& queryKeypoints, const vector<KeyPoint>& trainKeypoints, const vector<DMatch>& matches,
	Mat& homographyOut, vector<DMatch> inliersOut, vector<unsigned char> inliersMaskOut,
	float reprojectionThreshold, size_t minNumberMatchesAllowed) {
	
	if (matches.size() < minNumberMatchesAllowed) { return false; }
	
	// Prepare data for cv::findHomography
	vector<Point2f> srcPoints(matches.size());
	vector<Point2f> dstPoints(matches.size());
	for (size_t i = 0; i < matches.size(); i++) {
		srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
		dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
	}

	// Find homography matrix and get inliers mask
	inliersMaskOut.clear();
	inliersMaskOut.resize(srcPoints.size(), 0);	
	homographyOut = cv::findHomography(srcPoints, dstPoints, CV_FM_RANSAC, reprojectionThreshold, inliersMaskOut);
		
	for (size_t i = 0; i < inliersMaskOut.size(); ++i) {
		if (inliersMaskOut[i] > 0)
			inliersOut.push_back(matches[i]);
	}

	return inliersOut.size() >= minNumberMatchesAllowed;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
