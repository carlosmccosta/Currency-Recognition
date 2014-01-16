#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <vector>
#include <string>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

// project includes
#include "../Configs.h"
#include "DetectorEvaluationResult.h"
#include "../libs/Transformations/Transformations.h"

// namespace specific imports to avoid namespace pollution
using std::vector;
using std::string;
using std::stringstream;


using cv::Mat;
using cv::Ptr;
using cv::Vec3b;
using cv::Vec4i;
using cv::Point;
using cv::Point2f;
using cv::KeyPoint;
using cv::DMatch;
using cv::DescriptorMatcher;
using cv::Scalar;
using cv::Rect;
using cv::RotatedRect;
using cv::FileStorage;
using cv::imread;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
namespace ImageUtils {
	bool loadBinaryMask(const string& imagePath, Mat& binaryMaskOut);
	void loadImageMasks(const string& imagePath, vector<Mat>& masks);
	void retriveTargetsMasks(const string& imagePath, vector<Mat>& masksOut, const Scalar& lowerRange = Scalar(0, 0, 254), const Scalar& higherRange = Scalar(0, 0, 255));
	bool mergeTargetMasks(vector<Mat>& masks, Mat& mergedMaskOut);

	void splitKeyPoints(const string& imagePath, const vector<KeyPoint>& keypoints, vector< vector <KeyPoint> >& keypointsTargetClassOut, vector<KeyPoint>& keypointsNonTargetClassOut);

	void correctBoundingBox(Rect& boundingBoxInOut, int imageWidth, int imageHeight);
	void findMaskBoundingRectangles(Mat& mask, vector<Rect>& targetsBoundingRectanglesOut);
	
	bool loadMatrix(const string& filename, const string& tag, Mat& matrixOut);
	bool saveMatrix(const string& filename, const string& tag, const Mat& matrix);	

	bool matchDescriptorsWithRatioTest(Ptr<DescriptorMatcher> descriptorMatcher, const Mat& descriptorsQueryImage, const Mat& targetDescriptors, vector<DMatch>& matchesFilteredOut, float maxDistanceRatio = 0.75f);
	bool refineMatchesWithHomography(const vector<KeyPoint>& queryKeypoints, const vector<KeyPoint>& trainKeypoints, const vector<DMatch>& matches,
		Mat& homographyOut, vector<DMatch>& inliersOut, vector<unsigned char>& inliersMaskOut,
		float reprojectionThreshold = 3.0f, double confidence = 0.995, int maxIters = 5000, size_t minNumberMatchesAllowed = 4);

	void removeInliersFromKeypointsAndDescriptors(const vector<DMatch>& inliers, vector<KeyPoint>& keypointsQueryImageInOut, Mat& descriptorsQueryImageInOut);

	void drawContour(Mat& image, const vector<Point>& contour, const Scalar& color = Scalar(255, 255, 255), int thickness = 2);
	double computeContourAspectRatio(const vector<Point>& contour);
	double computeContourCircularity(const vector<Point>& contour);

	string getFilenameWithoutExtension(const string& filepath);
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
