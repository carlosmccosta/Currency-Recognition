#include "ImageDetector.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageDetector::ImageDetector(Ptr<FeatureDetector> featureDetector, Ptr<DescriptorExtractor> descriptorExtractor, Ptr<DescriptorMatcher> descriptorMatcher, Ptr<ImagePreprocessor> imagePreprocessor,
	const string& configurationTags, const string& selectorTags, const vector<string>& referenceImagesDirectories,
	bool useInliersGlobalMatch,
	const string& referenceImagesListPath, const string& testImagesListPath) :
	_featureDetector(featureDetector), _descriptorExtractor(descriptorExtractor), _descriptorMatcher(descriptorMatcher),
	_imagePreprocessor(imagePreprocessor), _configurationTags(configurationTags), _selectorTags(selectorTags),
	_referenceImagesDirectories(referenceImagesDirectories), _referenceImagesListPath(referenceImagesListPath), _testImagesListPath(testImagesListPath),
	_contourAspectRatioRange(-1, -1), _contourCircularityRange(-1, -1) {
	
	setupTargetDB(referenceImagesListPath, useInliersGlobalMatch);
	setupTargetsShapesRanges();
}


ImageDetector::~ImageDetector() {}


bool ImageDetector::setupTargetDB(const string& referenceImagesListPath, bool useInliersGlobalMatch) {
	_targetDetectors.clear();

	ifstream imgsList(referenceImagesListPath);
	if (imgsList.is_open()) {
		string configurationLine;
		vector<string> configurations;
		while (getline(imgsList, configurationLine)) {
			configurations.push_back(configurationLine);
		}
		int numberOfFiles = configurations.size();


		cout << "    -> Initializing recognition database with " << numberOfFiles << " reference images and with " << _referenceImagesDirectories.size() << " levels of detail..." << endl;
		PerformanceTimer performanceTimer;
		performanceTimer.start();

		//#pragma omp parallel for schedule(dynamic)
		for (int configIndex = 0; configIndex < numberOfFiles; ++configIndex) {			
			string filename;
			size_t targetTag;
			string separator;
			Scalar color;			
			stringstream ss(configurations[configIndex]);
			ss >> filename >> separator >> targetTag >> separator >> color[2] >> color[1] >> color[0];

			TargetDetector targetDetector(_featureDetector, _descriptorExtractor, _descriptorMatcher, targetTag, color, useInliersGlobalMatch);

			for (size_t i = 0; i < _referenceImagesDirectories.size(); ++i) {
				string referenceImagesDirectory = _referenceImagesDirectories[i];
				Mat targetImage;

				stringstream referenceImgePath;
				referenceImgePath << REFERENCE_IMGAGES_DIRECTORY << referenceImagesDirectory << "/" << filename;
				cout << "     => Adding reference image " << referenceImgePath.str() << endl;
				if (_imagePreprocessor->loadAndPreprocessImage(referenceImgePath.str(), targetImage, CV_LOAD_IMAGE_GRAYSCALE, false)) {
					string filenameWithoutExtension = ImageUtils::getFilenameWithoutExtension(filename);					
					stringstream maskFilename;
					maskFilename << REFERENCE_IMGAGES_DIRECTORY << referenceImagesDirectory << "/" << filenameWithoutExtension << MASK_TOKEN << MASK_EXTENSION;

					Mat targetROIs;
					if (ImageUtils::loadBinaryMask(maskFilename.str(), targetROIs)) {						
						targetDetector.setupTargetRecognition(targetImage, targetROIs);					

						vector<KeyPoint>& targetKeypoints = targetDetector.getTargetKeypoints();
						stringstream imageKeypointsFilename;
						imageKeypointsFilename << REFERENCE_IMGAGES_ANALYSIS_DIRECTORY << filenameWithoutExtension << "_" << referenceImagesDirectory << _selectorTags << IMAGE_OUTPUT_EXTENSION;
						if (targetKeypoints.empty()) {
							imwrite(imageKeypointsFilename.str(), targetImage);
						} else {
							Mat imageKeypoints;
							cv::drawKeypoints(targetImage, targetKeypoints, imageKeypoints, TARGET_KEYPOINT_COLOR);
							imwrite(imageKeypointsFilename.str(), imageKeypoints);
						}
					}					
				}
			}

			//#pragma omp critical
			_targetDetectors.push_back(targetDetector);
		}

		cout << "    -> Finished initialization of targets database in " << performanceTimer.getElapsedTimeFormated() << "\n" << endl;

		return !_targetDetectors.empty();
	} else {
		return false;
	}
}


void ImageDetector::setupTargetsShapesRanges(const string& maskPath) {
	Mat shapeROIs;
	if (ImageUtils::loadBinaryMask(maskPath, shapeROIs)) {
		vector< vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(shapeROIs, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		int contoursSize = (int)contours.size();

		#pragma omp parallel for
		for (int i = 0; i < contoursSize; ++i) {
			double contourAspectRatio = ImageUtils::computeContourAspectRatio(contours[i]);
			double contourCircularity = ImageUtils::computeContourCircularity(contours[i]);

			#pragma omp critical
			if (_contourAspectRatioRange[0] == -1 || contourAspectRatio < _contourAspectRatioRange[0]) {				
				_contourAspectRatioRange[0] = contourAspectRatio;
			}

			#pragma omp critical
			if (_contourAspectRatioRange[1] == -1 || contourAspectRatio > _contourAspectRatioRange[1]) {				
				_contourAspectRatioRange[1] = contourAspectRatio;
			}

			#pragma omp critical
			if (_contourCircularityRange[0] == -1 || contourCircularity < _contourCircularityRange[0]) {				
				_contourCircularityRange[0] = contourCircularity;
			}

			#pragma omp critical
			if (_contourCircularityRange[1] == -1 || contourCircularity > _contourCircularityRange[1]) {				
				_contourCircularityRange[1] = contourCircularity;
			}
		}		
	}
}


Ptr< vector< Ptr<DetectorResult> > > ImageDetector::detectTargets(Mat& image, float minimumMatchAllowed, float minimumTargetAreaPercentage,
	float maxDistanceRatio, float reprojectionThresholdPercentage, double confidence, int maxIters, size_t minimumNumberInliers) {
	Ptr< vector< Ptr<DetectorResult> > > detectorResults(new vector< Ptr<DetectorResult> >());

	vector<KeyPoint> keypointsQueryImage;
	_featureDetector->detect(image, keypointsQueryImage);
	if (keypointsQueryImage.size() < 4) { return detectorResults; }

	Mat descriptorsQueryImage;
	_descriptorExtractor->compute(image, keypointsQueryImage, descriptorsQueryImage);

	cv::drawKeypoints(image, keypointsQueryImage, image, NONTARGET_KEYPOINT_COLOR);

	float bestMatch = 0;
	Ptr<DetectorResult> bestDetectorResult;

	int targetDetectorsSize = _targetDetectors.size();
	bool validDetection = true;
	float reprojectionThreshold = image.cols * reprojectionThresholdPercentage;
	//float reprojectionThreshold = 3.0;

	do {
		bestMatch = 0;

		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < targetDetectorsSize; ++i) {
			_targetDetectors[i].updateCurrentLODIndex(image);
			Ptr<DetectorResult> detectorResult = _targetDetectors[i].analyzeImage(keypointsQueryImage, descriptorsQueryImage, maxDistanceRatio, reprojectionThreshold, confidence, maxIters, minimumNumberInliers);
			if (detectorResult->getBestROIMatch() > minimumMatchAllowed) {
				float contourArea = (float)cv::contourArea(detectorResult->getTargetContour());
				float imageArea = (float)(image.cols * image.rows);
				float contourAreaPercentage = contourArea / imageArea;

				if (contourAreaPercentage > minimumTargetAreaPercentage) {
					double contourAspectRatio = ImageUtils::computeContourAspectRatio(detectorResult->getTargetContour());
					if (contourAspectRatio > _contourAspectRatioRange[0] && contourAspectRatio < _contourAspectRatioRange[1]) {
						double contourCircularity = ImageUtils::computeContourCircularity(detectorResult->getTargetContour());
						if (contourCircularity > _contourCircularityRange[0] && contourCircularity < _contourCircularityRange[1]) {
							if (cv::isContourConvex(detectorResult->getTargetContour())) {
								#pragma omp critical
								{
									if (detectorResult->getBestROIMatch() > bestMatch) {
										bestMatch = detectorResult->getBestROIMatch();
										bestDetectorResult = detectorResult;
									}
								}
							}
						}
					}
				}
			}
		}

		validDetection = bestMatch > minimumMatchAllowed && bestDetectorResult->getInliers().size() > minimumNumberInliers;

		if (bestDetectorResult.obj != NULL && validDetection) {			
			detectorResults->push_back(bestDetectorResult);			

			// remove inliers of best match to detect more occurrences of targets
			ImageUtils::removeInliersFromKeypointsAndDescriptors(bestDetectorResult->getInliers(), keypointsQueryImage, descriptorsQueryImage);
		}		
	} while (validDetection);

	return detectorResults;
}


vector<size_t> ImageDetector::detectTargetsAndOutputResults(Mat& image, const string& imageFilename, bool useHighGUI) {
	Mat imageBackup = image.clone();
	Ptr< vector< Ptr<DetectorResult> > > detectorResultsOut = detectTargets(image);
	vector<size_t> results;		

	stringstream imageInliersOutputFilename;
	imageInliersOutputFilename << TEST_OUTPUT_DIRECTORY << imageFilename << FILENAME_SEPARATOR << _configurationTags << FILENAME_SEPARATOR << INLIERS_MATCHES << FILENAME_SEPARATOR;

	for (size_t i = 0; i < detectorResultsOut->size(); ++i) {		
		Ptr<DetectorResult> detectorResult = (*detectorResultsOut)[i];		
		results.push_back(detectorResult->getTargetValue());

		cv::drawKeypoints(image, detectorResult->getInliersKeypoints(), image, TARGET_KEYPOINT_COLOR);		

		stringstream ss;
		ss << detectorResult->getTargetValue();

		Mat imageMatchesSingle = imageBackup.clone();		
		Mat matchesInliers = detectorResult->getInliersMatches(imageMatchesSingle);

		try {
			Rect boundingBox = cv::boundingRect(detectorResult->getTargetContour());
			ImageUtils::correctBoundingBox(boundingBox, image.cols, image.rows);
			GUIUtils::drawLabelInCenterOfROI(ss.str(), image, boundingBox);
			GUIUtils::drawLabelInCenterOfROI(ss.str(), matchesInliers, boundingBox);
			ImageUtils::drawContour(image, detectorResult->getTargetContour(), detectorResult->getContourColor());
			ImageUtils::drawContour(matchesInliers, detectorResult->getTargetContour(), detectorResult->getContourColor());
		} catch (...) {
			std::cerr << "!!! Drawing outside image !!!" << endl;
		}

		if (useHighGUI) {
			stringstream windowName;
			windowName << "Target inliers matches (window " << i << ")";
			cv::namedWindow(windowName.str(), CV_WINDOW_KEEPRATIO);
			cv::imshow(windowName.str(), matchesInliers);
			cv::waitKey(10);
		}

		stringstream imageOutputFilenameFull;
		imageOutputFilenameFull << imageInliersOutputFilename.str() << i << IMAGE_OUTPUT_EXTENSION;
		imwrite(imageOutputFilenameFull.str(), matchesInliers);				
	}	

	sort(results.begin(), results.end());

	cout << "    -> Detected " << results.size() << (results.size() != 1 ? " targets" : " target");
	size_t globalResult = 0;
	stringstream resultsSS;
	if (!results.empty()) {
		resultsSS << " (";
		for (size_t i = 0; i < results.size(); ++i) {
			size_t resultValue = results[i];
			resultsSS << " " << resultValue;
			globalResult += resultValue;
		}
		resultsSS << " )";
		cout << resultsSS.str();
	}
	cout << endl;

	stringstream globalResultSS;
	globalResultSS << "Global result: " << globalResult << resultsSS.str();
	Rect globalResultBoundingBox(0, 0, image.cols, image.rows);
	GUIUtils::drawImageLabel(globalResultSS.str(), image, globalResultBoundingBox);

	stringstream imageOutputFilename;
	imageOutputFilename << TEST_OUTPUT_DIRECTORY << imageFilename << FILENAME_SEPARATOR << _configurationTags << IMAGE_OUTPUT_EXTENSION;
	imwrite(imageOutputFilename.str(), image);

	return results;
}


DetectorEvaluationResult ImageDetector::evaluateDetector(const string& testImgsList, bool saveResults) {
	double globalPrecision = 0;
	double globalRecall = 0;
	double globalAccuracy = 0;
	size_t numberTestImages = 0;

	stringstream resultsFilename;
	resultsFilename << TEST_OUTPUT_DIRECTORY << _configurationTags << FILENAME_SEPARATOR << RESULTS_FILE;
	ofstream resutlsFile(resultsFilename.str());

	ifstream imgsList(testImgsList);
	if (resutlsFile.is_open() && imgsList.is_open()) {
		resutlsFile << RESULTS_FILE_HEADER << "\n" << endl;

		string filename;
		vector<string> imageFilenames;
		vector< vector<size_t> > expectedResults;
		while (getline(imgsList, filename)) {											
			imageFilenames.push_back(filename);

			vector<size_t> expectedResultFromTest;
			extractExpectedResultsFromFilename(filename, expectedResultFromTest);
			expectedResults.push_back(expectedResultFromTest);
		}
		int numberOfTests = imageFilenames.size();

		cout << "    -> Evaluating detector with " << numberOfTests << " test images..." << endl;
		PerformanceTimer globalPerformanceTimer;
		globalPerformanceTimer.start();

		//#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < numberOfTests; ++i) {			
			PerformanceTimer testPerformanceTimer;
			testPerformanceTimer.start();

			string imageFilename = imageFilenames[i];
			//string imageFilename = ImageUtils::getFilenameWithoutExtension("");
			string imageFilenameWithPath = TEST_IMGAGES_DIRECTORY + imageFilenames[i];
			stringstream detectorEvaluationResultSS;
			DetectorEvaluationResult detectorEvaluationResult;
			Mat imagePreprocessed;
			cout << "\n    -> Evaluating image " << imageFilename << " (" << (i + 1) << "/" << numberOfTests << ")" << endl;
			if (_imagePreprocessor->loadAndPreprocessImage(imageFilenameWithPath, imagePreprocessed, CV_LOAD_IMAGE_GRAYSCALE, false)) {				
				vector<size_t> results = detectTargetsAndOutputResults(imagePreprocessed, imageFilename, false);				

				detectorEvaluationResult = DetectorEvaluationResult(results, expectedResults[i]);
				globalPrecision += detectorEvaluationResult.getPrecision();
				globalRecall += detectorEvaluationResult.getRecall();
				globalAccuracy += detectorEvaluationResult.getAccuracy();

				detectorEvaluationResultSS << PRECISION_TOKEN << ": " << detectorEvaluationResult.getPrecision() << " | " << RECALL_TOKEN << ": " << detectorEvaluationResult.getRecall() << " | " << ACCURACY_TOKEN << ": " << detectorEvaluationResult.getAccuracy();

				++numberTestImages;

				if (saveResults) {					
					resutlsFile << imageFilename << " -> " << detectorEvaluationResultSS.str() << endl;
				}
			}
			cout << "    -> Evaluation of image " << imageFilename << " finished in " << testPerformanceTimer.getElapsedTimeFormated() << endl;
			cout << "    -> " << detectorEvaluationResultSS.str() << endl;
		}

		globalPrecision /= (double)numberTestImages;
		globalRecall /= (double)numberTestImages;
		globalAccuracy /= (double)numberTestImages;

		stringstream detectorEvaluationGloablResultSS;
		detectorEvaluationGloablResultSS << GLOBAL_PRECISION_TOKEN << ": " << globalPrecision << " | " << GLOBAL_RECALL_TOKEN << ": " << globalRecall << " | " << GLOBAL_ACCURACY_TOKEN << ": " << globalAccuracy;

		resutlsFile << "\n\n" << RESULTS_FILE_FOOTER << endl;
		resutlsFile << " ==> " << detectorEvaluationGloablResultSS.str() << endl;
		cout << "\n    -> Finished evaluation of detector in " << globalPerformanceTimer.getElapsedTimeFormated() << " || " << detectorEvaluationGloablResultSS.str() << "\n" << endl;
	}

	return DetectorEvaluationResult(globalPrecision, globalRecall, globalAccuracy);
}


void ImageDetector::extractExpectedResultsFromFilename(string filename, vector<size_t>& expectedResultFromTestOut) {
	for (size_t i = 0; i < filename.size(); ++i) {
		char letter = filename[i];
		if (letter == '-') {
			filename[i] = ' ';
		} else if (letter == '.' || letter == '_') {
			filename = filename.substr(0, i);
			break;
		}
	}
	
	stringstream ss(filename);
	size_t number;	
	while (ss >> number) {
		expectedResultFromTestOut.push_back(number);
	}
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </ImageDetector>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
