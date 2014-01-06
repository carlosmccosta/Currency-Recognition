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
#include "DetectorEvaluationResult.h"
#include "../Configs.h"

// namespace specific imports to avoid namespace pollution
using std::vector;
using std::string;
using std::stringstream;


using cv::Mat;
using cv::Vec3b;
using cv::Vec4i;
using cv::Point;
using cv::Point2f;
using cv::KeyPoint;
using cv::DMatch;
using cv::Scalar;
using cv::Rect;
using cv::FileStorage;
using cv::imread;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
namespace ImageUtils {
	void loadImageMasks(string imagePath, vector<Mat>& masks);
	void retriveTargetsMasks(string imagePath, vector<Mat>& masksOut, Scalar lowerRange = Scalar(0, 0, 254), Scalar higherRange = Scalar(0, 0, 255));
	bool mergeTargetMasks(vector<Mat>& masks, Mat& mergedMaskOut);

	void splitKeyPoints(string imagePath, const vector<KeyPoint>& keypoints, vector< vector <KeyPoint> >& keypointsTargetClass, vector<KeyPoint>& keypointsNonTargetClass);

	void correctBoundingBox(Rect& boundingBox, int imageWidth, int imageHeight);
	void findMaskBoundingRectangles(Mat& mask, vector<Rect>& targetsBoundingRectangles);
	
	bool loadMatrix(string filename, string tag, Mat& matrixOut);
	bool saveMatrix(string filename, string tag, const Mat& matrix);

	bool refineMatchesWithHomography(const vector<KeyPoint>& queryKeypoints, const vector<KeyPoint>& trainKeypoints, const vector<DMatch>& matches,
		Mat& homographyOut, vector<DMatch>& inliersOut, vector<unsigned char>& inliersMaskOut,
		float reprojectionThreshold = 3.0f, size_t minNumberMatchesAllowed = 4);

	void drawContour(Mat& image, vector<Point2f> contour, Scalar color = Scalar(255,255,255), int thickness = 2);

	string getFilenameWithoutExtension(string filepath);
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
