#include "DetectorEvaluationResult.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
DetectorEvaluationResult::DetectorEvaluationResult() {}

DetectorEvaluationResult::DetectorEvaluationResult(double precision, double recall, double accuracy) :
_precision(precision), _recall(recall), _accuracy(accuracy) {}

DetectorEvaluationResult::DetectorEvaluationResult(size_t truePositives, size_t trueNegatives, size_t falsePositives, size_t falseNegatives) :
	_truePositives(truePositives), _trueNegatives(trueNegatives), _falsePositives(falsePositives), _falseNegatives(falseNegatives) {	

	updateMeasures();
}

DetectorEvaluationResult::DetectorEvaluationResult(vector<size_t> results, vector<size_t> expectedResults) :
	_truePositives(0), _trueNegatives(0), _falsePositives(0), _falseNegatives(0) {
	std::sort(results.begin(), results.end());
	std::sort(expectedResults.begin(), expectedResults.end());

	for (size_t resultsIndex = 0; resultsIndex < results.size(); ++resultsIndex) {
		vector<size_t>::iterator it = std::find(expectedResults.begin(), expectedResults.end(), results[resultsIndex]);

		if (it != expectedResults.end()) {
			++_truePositives;
			expectedResults.erase(it);
		} else {
			++_falsePositives;
		}
	}
	
	_falseNegatives = expectedResults.size();

	updateMeasures();
}

DetectorEvaluationResult::DetectorEvaluationResult(Mat& votingMask, vector<Mat>& targetMasks, unsigned short votingMaskThreshold) :
	_truePositives(0), _trueNegatives(0), _falsePositives(0), _falseNegatives(0) {
	Mat mergedTargetsMask;
	if (ImageUtils::mergeTargetMasks(targetMasks, mergedTargetsMask)) {
		computeMasksSimilarity(votingMask, mergedTargetsMask, votingMaskThreshold, &_truePositives, &_trueNegatives, &_falsePositives, &_falseNegatives);

		updateMeasures();
	}
}


bool DetectorEvaluationResult::computeMasksSimilarity(Mat& votingMask, Mat& mergedTargetsMask, unsigned short votingMaskThreshold,
	size_t* truePositivesOut, size_t* trueNegativesOut, size_t* falsePositivesOut, size_t* falseNegativesOut) {
	if (votingMask.rows == mergedTargetsMask.rows && votingMask.cols == mergedTargetsMask.cols) {
		size_t truePositives = 0;
		size_t trueNegatives = 0;
		size_t falsePositives = 0;
		size_t falseNegatives = 0;

		#pragma omp parallel for schedule(dynamic)
		for (int votingMaskY = 0; votingMaskY < votingMask.rows; ++votingMaskY) {
			for (int votingMaskX = 0; votingMaskX < votingMask.cols; ++votingMaskX) {
				if (votingMask.at<unsigned short>(votingMaskY, votingMaskX) > votingMaskThreshold) {
					if (mergedTargetsMask.at<unsigned char>(votingMaskY, votingMaskX) > 0) {
						#pragma omp atomic
						++truePositives;
					} else {
						#pragma omp atomic
						++falsePositives;
					}
				} else {
					if (mergedTargetsMask.at<unsigned char>(votingMaskY, votingMaskX) > 0) {
						#pragma omp atomic
						++falseNegatives;
					} else {
						#pragma omp atomic
						++trueNegatives;
					}
				}
			}
		}

		*truePositivesOut = truePositives;
		*trueNegativesOut = trueNegatives;
		*falsePositivesOut = falsePositives;
		*falseNegativesOut = falseNegatives;
		return true;
	}

	return false;
}


double DetectorEvaluationResult::computePrecision(size_t truePositives, size_t falsePositives) {
	double divisor = truePositives + falsePositives;
	if (divisor == 0) {
		return 0;
	}

	return (double)truePositives / divisor;
}


double DetectorEvaluationResult::computeRecall(size_t truePositives, size_t falseNegatives) {
	double divisor = truePositives + falseNegatives;
	if (divisor == 0) {
		return 0;
	}

	return (double)truePositives / divisor;
}


double DetectorEvaluationResult::computeAccuracy(size_t truePositives, size_t trueNegatives, size_t falsePositives, size_t falseNegatives) {
	double divisor = truePositives + trueNegatives + falsePositives + falseNegatives;
	if (divisor == 0) {
		return 0;
	}

	return (double)(truePositives + trueNegatives) / divisor;
}


void DetectorEvaluationResult::updateMeasures() {
	_precision = computePrecision(_truePositives, _falsePositives);
	_recall = computeRecall(_truePositives, _falseNegatives);
	_accuracy = computeAccuracy(_truePositives, _trueNegatives, _falsePositives, _falseNegatives);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ClassifierEvaluationResult>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
