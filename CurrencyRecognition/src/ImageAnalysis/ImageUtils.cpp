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


void ImageUtils::retriveTargetsMasks(string imagePath, vector<Mat>& masks, Scalar lowerRange, Scalar higherRange) {
	loadImageMasks(imagePath, masks);
	int masksSize = masks.size();

	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < masksSize; ++i) {
		/*Mat& before = masks[i];
		Mat after;
		cv::inRange(before, lowerRange, higherRange, after);
		masks[i] = after;*/

		cv::inRange(masks[i], lowerRange, higherRange, masks[i]);
	}
}


bool ImageUtils::mergeTargetMasks(vector<Mat>& masks, Mat& mergedMask) {
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

	mergedMask = Mat::zeros(maxRows, maxCols, CV_8UC1);
	for (int i = 0; i < masksSize; ++i) {
		if (masks[i].rows == maxRows && masks[i].cols == maxCols) {
			cv::bitwise_or(mergedMask, masks[i], mergedMask);
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
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
