#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>

// project includes
#include "ImageUtils.h"


// namespace specific imports to avoid namespace pollution
using std::vector;

using cv::Mat;
using cv::Vec3b;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class DetectorEvaluationResult {
public:
	DetectorEvaluationResult();
	DetectorEvaluationResult(double precision, double recall, double accuracy);
	DetectorEvaluationResult(size_t truePositives, size_t trueNegatives, size_t falsePositives, size_t falseNegatives);	
	DetectorEvaluationResult(vector<size_t> results, vector<size_t> expectedResults);
	DetectorEvaluationResult(Mat& votingMask, vector<Mat>& targetMasks, unsigned short votingMaskThreshold = 1);
	virtual ~DetectorEvaluationResult() {}

	static bool computeMasksSimilarity(Mat& votingMask, Mat& mergedTargetsMask, unsigned short votingMaskThreshold,
		size_t* truePositivesOut, size_t* trueNegativesOut, size_t* falsePositivesOut, size_t* falseNegativesOut);

	static double computePrecision(size_t truePositives, size_t falsePositives);
	static double computeRecall(size_t truePositives, size_t falseNegatives);
	static double computeAccuracy(size_t truePositives, size_t trueNegatives, size_t falsePositives, size_t falseNegatives);

	void updateMeasures();

	// ------------------------------------------------------------------------------  <gets | sets> -------------------------------------------------------------------------------
	double getPrecision() const { return _precision; }
	void setPrecision(double val) { _precision = val; }
	double getRecall() const { return _recall; }
	void setRecall(double val) { _recall = val; }
	double getAccuracy() const { return _accuracy; }
	void setAccuracy(double val) { _accuracy = val; }

	size_t getTruePositives() const { return _truePositives; }
	void setTruePositives(size_t val) { _truePositives = val; }
	size_t getTrueNegatives() const { return _trueNegatives; }
	void setTrueNegatives(size_t val) { _trueNegatives = val; }
	size_t getFalsePositives() const { return _falsePositives; }
	void setFalsePositives(size_t val) { _falsePositives = val; }
	size_t getFalseNegatives() const { return _falseNegatives; }
	void setFalseNegatives(size_t val) { _falseNegatives = val; }
	// ------------------------------------------------------------------------------  </gets | sets> ------------------------------------------------------------------------------

private:
	double _precision;	// truePositives / (truePositives + falsePositives)
	double _recall;		// truePositives / (truePositives + falseNegatives)
	double _accuracy;	// (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
	
	size_t _truePositives;
	size_t _trueNegatives;
	size_t _falsePositives;
	size_t _falseNegatives;	
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
