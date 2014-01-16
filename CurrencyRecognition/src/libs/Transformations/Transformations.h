#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <copyright notice> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// This implementation resulted from the refactoring of code distributed in the OpenCV 2.4.8
// The refactoring was performed to allow the fine tunning of the findHomography function
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </copyright notice> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// OpenCV includes
#include <opencv2/core/core.hpp>

// project includes
#include "HomographyEstimator.h"

// namespace specific imports to avoid namespace pollution
using cv::Mat;
using cv::InputArray;
using cv::OutputArray;
using cv::Scalar;
using cv::Ptr;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Transformations>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
namespace Transformations {	
	Mat findHomography(InputArray srcPoints, InputArray dstPoints, int method = 0, double ransacReprojThreshold = 3, OutputArray mask = cv::noArray(), double confidence = 0.995, int maxIters = 5000);
	int findHomography(const CvMat* src_points, const CvMat* dst_points, CvMat* homography, int method = 0, double ransacReprojThreshold = 3, CvMat* mask = 0, double confidence = 0.995, int maxIters = 5000);

	template<typename T> int icvCompressPoints(T* ptr, const uchar* mask, int mstep, int count) {
		int i, j;
		for (i = j = 0; i < count; i++)
		if (mask[i*mstep]) {
			if (i > j)
				ptr[j] = ptr[i];
			j++;
		}
		return j;
	}
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </Transformations>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
