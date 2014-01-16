#include "Transformations.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <copyright notice> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// This implementation resulted from the refactoring of code distributed in the OpenCV 2.4.8
// The refactoring was performed to allow the fine tunning of the findHomography function
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </copyright notice> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Transformations>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Mat Transformations::findHomography(InputArray _points1, InputArray _points2, int method, double ransacReprojThreshold, OutputArray _mask, double confidence, int maxIters) {
	Mat points1 = _points1.getMat(), points2 = _points2.getMat();
	int npoints = points1.checkVector(2);
	CV_Assert(npoints >= 0 && points2.checkVector(2) == npoints && points1.type() == points2.type());

	Mat H(3, 3, CV_64F);
	CvMat _pt1 = points1, _pt2 = points2;
	CvMat matH = H, c_mask, *p_mask = 0;
	if (_mask.needed()) {
		_mask.create(npoints, 1, CV_8U, -1, true);
		p_mask = &(c_mask = _mask.getMat());
	}

	bool ok = findHomography(&_pt1, &_pt2, &matH, method, ransacReprojThreshold, p_mask, confidence, maxIters) > 0;
	if (!ok) H = Scalar(0);

	return H;
}


int Transformations::findHomography(const CvMat* objectPoints, const CvMat* imagePoints, CvMat* __H, int method, double ransacReprojThreshold, CvMat* mask, double confidence, int maxIters) {	
	const double defaultRANSACReprojThreshold = 3;
	bool result = false;
	Ptr<CvMat> m, M, tempMask;

	double H[9];
	CvMat matH = cvMat(3, 3, CV_64FC1, H);
	int count;

	CV_Assert(CV_IS_MAT(imagePoints) && CV_IS_MAT(objectPoints));

	count = MAX(imagePoints->cols, imagePoints->rows);
	CV_Assert(count >= 4);
	if (ransacReprojThreshold <= 0)
		ransacReprojThreshold = defaultRANSACReprojThreshold;

	m = cvCreateMat(1, count, CV_64FC2);
	cvConvertPointsHomogeneous(imagePoints, m);

	M = cvCreateMat(1, count, CV_64FC2);
	cvConvertPointsHomogeneous(objectPoints, M);

	if (mask)
		CV_Assert(CV_IS_MASK_ARR(mask) && CV_IS_MAT_CONT(mask->type) && (mask->rows == 1 || mask->cols == 1) && mask->rows*mask->cols == count);	

	if (mask || count > 4)
		tempMask = cvCreateMat(1, count, CV_8U);

	if (!tempMask.empty())
		cvSet(tempMask, cvScalarAll(1.));

	HomographyEstimator estimator(4);
	if (count == 4)
		method = 0;

	if (method == CV_LMEDS)
		result = estimator.runLMeDS(M, m, &matH, tempMask, confidence, maxIters);
	else if (method == CV_RANSAC)
		result = estimator.runRANSAC(M, m, &matH, tempMask, ransacReprojThreshold, confidence, maxIters);
	else
		result = estimator.runKernel(M, m, &matH) > 0;

	if (result && count > 4) {
		icvCompressPoints((CvPoint2D64f*)M->data.ptr, tempMask->data.ptr, 1, count);
		count = icvCompressPoints((CvPoint2D64f*)m->data.ptr, tempMask->data.ptr, 1, count);
		M->cols = m->cols = count;
		
		if (method == CV_RANSAC)
			estimator.runKernel(M, m, &matH);
		
		estimator.refine(M, m, &matH, 10);
	}

	if (result)
		cvConvert(&matH, __H);

	if (mask && tempMask) {
		if (CV_ARE_SIZES_EQ(mask, tempMask))
			cvCopy(tempMask, mask);
		else
			cvTranspose(tempMask, mask);
	}

	return (int)result;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </Transformations>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
