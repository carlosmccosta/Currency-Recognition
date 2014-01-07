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


void ImageUtils::correctBoundingBox(Rect& boundingBox, int imageWidth, int imageHeight) {
	if (boundingBox.x < 0) {
		boundingBox.width += boundingBox.x;
		boundingBox.x = 0;
	}

	if (boundingBox.x > imageWidth) {
		boundingBox.width = 0;
		boundingBox.x = imageWidth;
	}

	if (boundingBox.y < 0) {
		boundingBox.height += boundingBox.y;
		boundingBox.y = 0;
	}

	if (boundingBox.y > imageHeight) {
		boundingBox.height = 0;
		boundingBox.y = imageWidth;
	}

	int maxWidth = imageWidth - boundingBox.x;
	if (boundingBox.width > maxWidth) {
		boundingBox.width = maxWidth;
	}

	int maxHeight = imageHeight - boundingBox.y;
	if (boundingBox.height > maxHeight) {
		boundingBox.height = maxHeight;
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


bool ImageUtils::matchDescriptorsWithRatioTest(Ptr<DescriptorMatcher> descriptorMatcher, const Mat& descriptorsQueryImage, const Mat& targetDescriptors, vector<DMatch>& matchesFilteredOut, float maxDistanceRatio) {
	matchesFilteredOut.clear();
	vector< vector<DMatch> > matchesKNN;
	descriptorMatcher->knnMatch(descriptorsQueryImage, targetDescriptors, matchesKNN, 2);
	
	for (size_t matchPos = 0; matchPos < matchesKNN.size(); ++matchPos) {		
		if (matchesKNN[matchPos][0].distance <= maxDistanceRatio * matchesKNN[matchPos][1].distance) {
			matchesFilteredOut.push_back(matchesKNN[matchPos][0]);
		}
	}

	return !matchesFilteredOut.empty();
}


bool ImageUtils::refineMatchesWithHomography(const vector<KeyPoint>& queryKeypoints, const vector<KeyPoint>& trainKeypoints, const vector<DMatch>& matches,
	Mat& homographyOut, vector<DMatch>& inliersOut, vector<unsigned char>& inliersMaskOut,
	float reprojectionThreshold, size_t minNumberMatchesAllowed) {
	
	if (matches.size() < minNumberMatchesAllowed) { return false; }
	
	// Prepare data for cv::findHomography
	vector<Point2f> srcPoints(matches.size());
	vector<Point2f> dstPoints(matches.size());
	for (size_t i = 0; i < matches.size(); ++i) {
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

	return !inliersOut.empty();
}


void ImageUtils::drawContour(Mat& image, vector<Point2f> contour, Scalar color, int thickness) {
	for (size_t i = 0; i < contour.size(); ++i) {
		Point p1 = contour[i];
		Point p2;

		if (i == contour.size() - 1) {
			p2 = contour[0];
		} else {
			p2 = contour[i + 1];
		}

		try {
			cv::line(image, p1, p2, color, thickness);
		} catch (...) {}
	}
}


string ImageUtils::getFilenameWithoutExtension(string filepath) {
	size_t dotPosition = filepath.rfind(".");
	if (dotPosition != string::npos) {
		return filepath.substr(0, dotPosition);
	} else {
		return filepath;
	}
}


void ImageUtils::removeInliersFromKeypointsAndDescriptors(vector<DMatch>& inliers, vector<KeyPoint>& keypointsQueryImage, Mat& descriptorsQueryImage) {
	vector<int> inliersKeypointsPositions; // positions to remove

	for (size_t inlierIndex = 0; inlierIndex < inliers.size(); ++inlierIndex) {
		DMatch match = inliers[inlierIndex];
		inliersKeypointsPositions.push_back(match.queryIdx);
	}

	sort(inliersKeypointsPositions.begin(), inliersKeypointsPositions.end()); // must sort to delete from the end of vector in order to delete correct keypoints indexes
	
	/*for (int i = inliersKeypointsPositions.size() - 1; i >= 0; --i) {
		keypointsQueryImage.erase(keypointsQueryImage.begin() + inliersKeypointsPositions[i]);		
	}*/

	vector<KeyPoint> keypointsQueryImageBackup = keypointsQueryImage;
	keypointsQueryImage.clear();
	Mat filteredDescriptors;		
	for (int rowIndex = 0; rowIndex < descriptorsQueryImage.rows; ++rowIndex) {
		if (!binary_search(inliersKeypointsPositions.begin(), inliersKeypointsPositions.end(), rowIndex)) {
			keypointsQueryImage.push_back(keypointsQueryImageBackup[rowIndex]);
			filteredDescriptors.push_back(descriptorsQueryImage.row(rowIndex));			
		}
	}

	filteredDescriptors.copyTo(descriptorsQueryImage);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
