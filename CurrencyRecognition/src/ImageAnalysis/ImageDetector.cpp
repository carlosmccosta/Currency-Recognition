#include "ImageDetector.h"

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageDetector::ImageDetector() {}
ImageDetector::~ImageDetector() {}


void ImageDetector::detectTargets(Mat& image, vector<Rect>& targetsBoundingRectanglesOut, Mat& imageDetectionMasksOut, bool showTargetBoundingRectangles, bool showImageKeyPoints) {
	// TODO : targets detection
}


DetectorEvaluationResult ImageDetector::evaluateDetector(string testImgsList, bool saveResults) {
	double globalPrecision = 0;
	double globalRecall = 0;
	double globalAccuracy = 0;
	size_t numberTestImages = 0;

	stringstream resultsFilename;
	resultsFilename << TEST_OUTPUT_DIRECTORY << _configuration << FILENAME_SEPARATOR << RESULTS_FILE;
	ofstream resutlsFile(resultsFilename.str());

	ifstream imgsList(testImgsList);
	if (resutlsFile.is_open() && imgsList.is_open()) {
		resutlsFile << RESULTS_FILE_HEADER << "\n" << endl;

		string filename;
		vector<string> fileNames;
		while (getline(imgsList, filename)) {
			fileNames.push_back(filename);
		}
		int numberOfFiles = fileNames.size();

		cout << "    -> Evaluating detector with " << numberOfFiles << " test images..." << endl;
		PerformanceTimer performanceTimer;
		performanceTimer.start();

		//#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < numberOfFiles; ++i) {
			Mat imagePreprocessed;
			string imageFilename = IMGS_DIRECTORY + fileNames[i] + IMAGE_TOKEN;

			stringstream detectorEvaluationResultSS;
			DetectorEvaluationResult detectorEvaluationResult;
			cout << "\n    -> Evaluating image " << imageFilename << " (" << (i + 1) << "/" << numberOfFiles << ")" << endl;
			if (_imagePreprocessor.loadAndPreprocessImage(filename, imagePreprocessed, CV_LOAD_IMAGE_COLOR, false)) {
				vector<Rect> targetsBoundingRectangles;
				Mat imageDetectionMasksOut;
				detectTargets(imagePreprocessed, targetsBoundingRectangles, imageDetectionMasksOut, true, true);

				vector<Mat> masks;
				ImageUtils::retriveTargetsMasks(IMGS_DIRECTORY + fileNames[i], masks);

				detectorEvaluationResult = DetectorEvaluationResult(imageDetectionMasksOut, masks, 1);
				globalPrecision += detectorEvaluationResult.getPrecision();
				globalRecall += detectorEvaluationResult.getRecall();
				globalAccuracy += detectorEvaluationResult.getAccuracy();

				++numberTestImages;

				if (saveResults) {
					stringstream imageOutputFilename;
					imageOutputFilename << TEST_OUTPUT_DIRECTORY << fileNames[i] << FILENAME_SEPARATOR << _configuration;

					stringstream imageOutputFilenameMask;
					imageOutputFilenameMask << imageOutputFilename.str() << FILENAME_SEPARATOR << DETECTION_MASK << IMAGE_OUTPUT_EXTENSION;

					imageOutputFilename << IMAGE_OUTPUT_EXTENSION;

					imwrite(imageOutputFilename.str(), imagePreprocessed);
					imwrite(imageOutputFilenameMask.str(), imageDetectionMasksOut);

					detectorEvaluationResultSS << PRECISION_TOKEN << ": " << detectorEvaluationResult.getPrecision() << " | " << RECALL_TOKEN << ": " << detectorEvaluationResult.getRecall() << " | " << ACCURACY_TOKEN << ": " << detectorEvaluationResult.getAccuracy();
					resutlsFile << imageFilename << " -> " << detectorEvaluationResultSS.str() << endl;
				}
			}
			cout << "    -> Evaluation of image " << imageFilename << " finished" << endl;
			cout << "    -> " << detectorEvaluationResultSS.str() << endl;
		}

		globalPrecision /= (double)numberTestImages;
		globalRecall /= (double)numberTestImages;
		globalAccuracy /= (double)numberTestImages;

		stringstream detectorEvaluationGloablResultSS;
		detectorEvaluationGloablResultSS << GLOBAL_PRECISION_TOKEN << ": " << globalPrecision << " | " << GLOBAL_RECALL_TOKEN << ": " << globalRecall << " | " << GLOBAL_ACCURACY_TOKEN << ": " << globalAccuracy;

		resutlsFile << "\n\n" << RESULTS_FILE_FOOTER << endl;
		resutlsFile << " ==> " << detectorEvaluationGloablResultSS.str() << endl;
		cout << "\n    -> Finished evaluation of detector in " << performanceTimer.getElapsedTimeFormated() << " || " << detectorEvaluationGloablResultSS.str() << "\n" << endl;
	}

	return DetectorEvaluationResult(globalPrecision, globalRecall, globalAccuracy);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
