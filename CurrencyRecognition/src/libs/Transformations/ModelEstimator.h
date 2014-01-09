#pragma once

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <copyright notice> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// This implementation resulted from the refactoring of code distributed in the OpenCV 2.4.8
// The refactoring was performed to allow the fine tunning of the findHomography function
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </copyright notice> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <algorithm>
#include <iterator>
#include <limits>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/features2d/features2d.hpp"
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ModelEstimator>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class ModelEstimator {
	public:
		ModelEstimator(int _modelPoints, CvSize _modelSize, int _maxBasicSolutions);
		virtual ~ModelEstimator();

		virtual int runKernel(const CvMat* m1, const CvMat* m2, CvMat* model) = 0;
		virtual bool runLMeDS(const CvMat* m1, const CvMat* m2, CvMat* model, CvMat* mask, double confidence = 0.99, int maxIters = 2000);
		virtual bool runRANSAC(const CvMat* m1, const CvMat* m2, CvMat* model, CvMat* mask, double threshold, double confidence = 0.99, int maxIters = 2000);
		virtual bool refine(const CvMat*, const CvMat*, CvMat*, int) { return true; }
		virtual void setSeed(int64 seed);

	protected:
		virtual void computeReprojError(const CvMat* m1, const CvMat* m2, const CvMat* model, CvMat* error) = 0;
		virtual int findInliers(const CvMat* m1, const CvMat* m2, const CvMat* model, CvMat* error, CvMat* mask, double threshold);
		int cvRANSACUpdateNumIters(double p, double ep, int model_points, int max_iters);
		virtual bool getSubset(const CvMat* m1, const CvMat* m2, CvMat* ms1, CvMat* ms2, int maxAttempts = 1000);
		virtual bool checkSubset(const CvMat* ms1, int count);

		CvRNG rng;
		int modelPoints;
		CvSize modelSize;
		int maxBasicSolutions;
		bool checkPartialSubsets;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ModelEstimator>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
