#include "ImageUtils.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
bool ImageUtils::loadBinaryMask(const string& imagePath, Mat& binaryMaskOut) {
	binaryMaskOut = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
	if (binaryMaskOut.data) {
		cv::threshold(binaryMaskOut, binaryMaskOut, 250, 255, CV_THRESH_BINARY);
		return true;
	}

	return false;
}


void ImageUtils::loadImageMasks(const string& imagePath, vector<Mat>& masks) {
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


void ImageUtils::retriveTargetsMasks(const string& imagePath, vector<Mat>& masksOut, const Scalar& lowerRange, const Scalar& higherRange) {
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


void ImageUtils::splitKeyPoints(const string& imagePath, const vector<KeyPoint>& keypoints, vector< vector <KeyPoint> >& keypointsTargetClassOut, vector<KeyPoint>& keypointsNonTargetClassOut) {
	keypointsTargetClassOut.clear();
	keypointsNonTargetClassOut.clear();

	vector<Mat> masks;
	loadImageMasks(imagePath, masks);
	int keyPointsSize = keypoints.size();	

	keypointsTargetClassOut.resize(masks.size());

	#pragma omp parallel for schedule(dynamic)
	for (int keyPointPosition = 0; keyPointPosition < keyPointsSize; ++keyPointPosition) {
		bool keyPointIsNotCar = true;

		for (size_t maskPosition = 0; maskPosition < masks.size(); ++maskPosition) {									
			Vec3b maskColorInKeyPointPosition = masks[maskPosition].at<Vec3b>(keypoints[keyPointPosition].pt);
						
			if (maskColorInKeyPointPosition[2] == 255) {
				#pragma omp critical
				keypointsTargetClassOut[maskPosition].push_back(keypoints[keyPointPosition]);
				
				keyPointIsNotCar = false;
				break;
			}			
		}
		
		if (keyPointIsNotCar) {
			#pragma omp critical
			keypointsNonTargetClassOut.push_back(keypoints[keyPointPosition]);
		}
	}
}


void ImageUtils::correctBoundingBox(Rect& boundingBoxInOut, int imageWidth, int imageHeight) {
	if (boundingBoxInOut.x < 0) {
		boundingBoxInOut.width += boundingBoxInOut.x;
		boundingBoxInOut.x = 0;
	}

	if (boundingBoxInOut.x > imageWidth) {
		boundingBoxInOut.width = 0;
		boundingBoxInOut.x = imageWidth;
	}

	if (boundingBoxInOut.y < 0) {
		boundingBoxInOut.height += boundingBoxInOut.y;
		boundingBoxInOut.y = 0;
	}

	if (boundingBoxInOut.y > imageHeight) {
		boundingBoxInOut.height = 0;
		boundingBoxInOut.y = imageWidth;
	}

	int maxWidth = imageWidth - boundingBoxInOut.x;
	if (boundingBoxInOut.width > maxWidth) {
		boundingBoxInOut.width = maxWidth;
	}

	int maxHeight = imageHeight - boundingBoxInOut.y;
	if (boundingBoxInOut.height > maxHeight) {
		boundingBoxInOut.height = maxHeight;
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


bool ImageUtils::loadMatrix(const string& filename, const string& tag, Mat& matrixOut) {
	FileStorage fs;
	if (fs.open(filename, FileStorage::READ)) {		
		fs[tag] >> matrixOut;

		fs.release();
		return true;
	}

	return false;
}


bool ImageUtils::saveMatrix(const string& filename, const string& tag, const Mat& matrix) {
	FileStorage fs;
	if (fs.open(filename, FileStorage::WRITE)) {		
		fs << tag << matrix;

		fs.release();
		return true;
	}

	return false;
}


bool ImageUtils::matchDescriptorsWithRatioTest(Ptr<DescriptorMatcher> descriptorMatcher, const Mat& descriptorsQueryImage, const Mat& targetDescriptors, vector<DMatch>& matchesFilteredOut, float maxDistanceRatio) {
	if (targetDescriptors.rows < 4) {
		return false;
	}
	
	matchesFilteredOut.clear();
	vector< vector<DMatch> > matchesKNN;
	descriptorMatcher->knnMatch(descriptorsQueryImage, targetDescriptors, matchesKNN, 2);
	
	for (size_t matchPos = 0; matchPos < matchesKNN.size(); ++matchPos) {
		if (matchesKNN[matchPos].size() >= 2) {
			if (matchesKNN[matchPos][0].distance <= maxDistanceRatio * matchesKNN[matchPos][1].distance) {
				matchesFilteredOut.push_back(matchesKNN[matchPos][0]);
			}
		}
	}

	return !matchesFilteredOut.empty();
}


bool ImageUtils::refineMatchesWithHomography(const vector<KeyPoint>& queryKeypoints, const vector<KeyPoint>& trainKeypoints, const vector<DMatch>& matches,
	Mat& homographyOut, vector<DMatch>& inliersOut, vector<unsigned char>& inliersMaskOut,
	float reprojectionThreshold, double confidence, int maxIters, size_t minNumberMatchesAllowed) {
	
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
	homographyOut = Transformations::findHomography(srcPoints, dstPoints, CV_FM_RANSAC, reprojectionThreshold, inliersMaskOut, confidence, maxIters);
		
	for (size_t i = 0; i < inliersMaskOut.size(); ++i) {
		if (inliersMaskOut[i] > 0)
			inliersOut.push_back(matches[i]);
	}

	return (inliersOut.size() >= minNumberMatchesAllowed);
}


void ImageUtils::removeInliersFromKeypointsAndDescriptors(const vector<DMatch>& inliers, vector<KeyPoint>& keypointsQueryImageInOut, Mat& descriptorsQueryImageInOut) {
	vector<int> inliersKeypointsPositions; // positions to remove

	for (size_t inlierIndex = 0; inlierIndex < inliers.size(); ++inlierIndex) {
		DMatch match = inliers[inlierIndex];
		inliersKeypointsPositions.push_back(match.queryIdx);
	}

	sort(inliersKeypointsPositions.begin(), inliersKeypointsPositions.end()); // must sort to delete from the end of vector in order to delete correct keypoints indexes

	/*for (int i = inliersKeypointsPositions.size() - 1; i >= 0; --i) {
	keypointsQueryImage.erase(keypointsQueryImage.begin() + inliersKeypointsPositions[i]);
	}*/

	vector<KeyPoint> keypointsQueryImageBackup = keypointsQueryImageInOut;
	keypointsQueryImageInOut.clear();
	Mat filteredDescriptors;
	for (int rowIndex = 0; rowIndex < descriptorsQueryImageInOut.rows; ++rowIndex) {
		if (!binary_search(inliersKeypointsPositions.begin(), inliersKeypointsPositions.end(), rowIndex)) {
			keypointsQueryImageInOut.push_back(keypointsQueryImageBackup[rowIndex]);
			filteredDescriptors.push_back(descriptorsQueryImageInOut.row(rowIndex));
		}
	}

	filteredDescriptors.copyTo(descriptorsQueryImageInOut);
}


void ImageUtils::drawContour(Mat& image, const vector<Point>& contour, const Scalar& color, int thickness) {
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


double ImageUtils::computeContourAspectRatio(const vector<Point>& contour) {
	RotatedRect contourEllipse = cv::minAreaRect(contour);
	return contourEllipse.size.width / contourEllipse.size.height;
}


double ImageUtils::computeContourCircularity(const vector<Point>& contour) {
	double area = contourArea(contour);
	double perimeter = cv::arcLength(contour, true);

	if (perimeter != 0) {
		return (4.0 * CV_PI * area) / (perimeter * perimeter);
	}

	return 0;
}


string ImageUtils::getFilenameWithoutExtension(const string& filepath) {
	size_t dotPosition = filepath.rfind(".");
	if (dotPosition != string::npos) {
		return filepath.substr(0, dotPosition);
	} else {
		return filepath;
	}
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
