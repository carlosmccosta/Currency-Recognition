#include "CLI.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Commnd Line user Interface>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
void CLI::showConsoleHeader() {		
	cout << "####################################################################################################\n";
	cout << "  >>>                                    Currency recognition                                  <<<  \n";
	cout << "####################################################################################################\n\n";
}


void CLI::startInteractiveCLI() {
	int userOption = 1;
	string filename = "";
	int cameraDeviceNumber = 0;
	
	ConsoleInput::getInstance()->clearConsoleScreen();
	showConsoleHeader();
	
	int screenWidth = 1920; // ConsoleInput::getInstance()->getIntCin("  >> Screen width (used to arrange windows): ", "  => Width >= 100 !!!\n", 100);
	int screenHeight = 1080; // ConsoleInput::getInstance()->getIntCin("  >> Screen height (used to arrange windows): ", "  => Width >= 100 !!!\n", 100);
	bool optionsOneWindow = false; // ConsoleInput::getInstance()->getYesNoCin("  >> Use only one window for options trackbars? (Y/N): ");
	bool setupOfImageRecognitionDone = false;

	do {
		try {
			ConsoleInput::getInstance()->clearConsoleScreen();
			showConsoleHeader();
		
			if (setupOfImageRecognitionDone) {
				userOption = getUserOption();
				if (userOption == 1) {
					setupImageRecognition();
				} else if (userOption == 2) {
					_imageDetector->evaluateDetector(TEST_IMGAGES_LIST);
				} else {
					if (userOption == 3 || userOption == 4) {
						filename = "";
						do {
							cout << "  >> Path to file inside imgs\\testDB folder: ";
							filename = ConsoleInput::getInstance()->getLineCin();

							if (filename == "") {
								cerr << "  => File path can't be empty!\n" << endl;
							}
						} while (filename == "");
					} else if (userOption == 5) {
						cameraDeviceNumber = ConsoleInput::getInstance()->getIntCin("  >> Insert the camera device number to use (default: 0): ", "  => Camera device number must be >= 0 !!!\n", 0);
					}
				
					ImageAnalysis imageAnalysis(_imagePreprocessor, _imageDetector);
					imageAnalysis.setScreenWidth(screenWidth);
					imageAnalysis.setScreenHeight(screenHeight);
					imageAnalysis.setOptionsOneWindow(optionsOneWindow);					

					switch (userOption) {
						case 3: { if (!imageAnalysis.processImage(filename)) { cerr << "  => Failed to load image " << filename << "!" << endl; } break; }
						case 4: { if (!imageAnalysis.processVideo(filename)) { cerr << "  => Failed to load video " << filename << "!" << endl; } break; }
						case 5: { if (!imageAnalysis.processVideo(cameraDeviceNumber)) { cerr << "  => Failed to open camera " << cameraDeviceNumber << "!" << endl; } break; }
						default: break;
					}				
				}
			} else {
				setupImageRecognition();
				setupOfImageRecognitionDone = true;
			}

			if (userOption != 0) {
				cout << "\n\n" << endl;
				ConsoleInput::getInstance()->getUserInput();
			}
		} catch (...) {
			cerr << "\n\n\n!!!!! Caught unexpected exception !!!!!\n\n\n" << endl;
		}
	} while (userOption != 0);

	cout << "\n\n\n" << endl;
	showVersion();
	cout << "\n\n" << endl;
	ConsoleInput::getInstance()->getUserInput();
}


int CLI::getUserOption() {
	cout << " ## Menu:\n";
	cout << "   1 - Setup image recognition configuration\n";
	cout << "   2 - Evaluate detector\n";
	cout << "   3 - Test detector from image\n";
	cout << "   4 - Test detector from video\n";
	cout << "   5 - Test detector from camera\n";
	cout << "   0 - Exit\n";

	return ConsoleInput::getInstance()->getIntCin("\n >>> Option [0, 5]: ", "Select one of the options above!", 0, 6);
}


void CLI::setupImageRecognition() {
	cout << "\n\n ## Image recognition setup:\n" << endl;

	int imagesDBLevelOfDetailSelection = selectImagesDBLevelOfDetail();
	cout << "\n\n\n";
	int inliersSelectionMethod = selectInliersSelectionMethod();
	cout << "\n\n\n";
	int featureDetectorSelection = selectFeatureDetector();
	cout << "\n\n\n";
	int descriptorExtractorSelection = selectDescriptorExtractor();
	cout << "\n\n\n";
	int descriptorMatcherSelection = selectDescriptorMatcher();
	cout << "\n\n\n";	

	Ptr<FeatureDetector> featureDetector;
	Ptr<DescriptorExtractor> descriptorExtractor;
	Ptr<DescriptorMatcher> descriptorMatcher;
	stringstream configurationTags;
	stringstream selectorTags;
	bool inliersSelectionMethodFlagToUseGlobalMatch = true;

	switch (featureDetectorSelection) {
		case 1: { featureDetector = new cv::SiftFeatureDetector();			configurationTags << "_SIFT-Detector"; selectorTags << "_SIFT-Detector"; break; }
		case 2: { featureDetector = new cv::SurfFeatureDetector(400);		configurationTags << "_SURF-Detector"; selectorTags << "_SURF-Detector"; break; }
		case 3: { featureDetector = new cv::GoodFeaturesToTrackDetector();	configurationTags << "_GFTT-Detector"; selectorTags << "_GFTT-Detector"; break; }
		case 4: { featureDetector = new cv::FastFeatureDetector();			configurationTags << "_FAST-Detector"; selectorTags << "_FAST-Detector"; break; }
		case 5: { featureDetector = new cv::OrbFeatureDetector();			configurationTags << "_ORB-Detector";  selectorTags << "_ORB-Detector"; break; }
		case 6: { featureDetector = new cv::BRISK();						configurationTags << "_BRISK-Detector"; selectorTags << "_BRISK-Detector"; break; }
		case 7: { featureDetector = new cv::StarFeatureDetector();			configurationTags << "_STAR-Detector"; selectorTags << "_STAR-Detector"; break; }
		case 8: { featureDetector = new cv::MserFeatureDetector();			configurationTags << "_MSER-Detector"; selectorTags << "_MSER-Detector"; break; }
		default: break;
	}

	switch (descriptorExtractorSelection) {
		case 1: { descriptorExtractor = new cv::SiftDescriptorExtractor();	configurationTags << "_SIFT-Extractor"; break; }
		case 2: { descriptorExtractor = new cv::SurfDescriptorExtractor();	configurationTags << "_SURF-Extractor"; break; }
		case 3: { descriptorExtractor = new cv::FREAK();					configurationTags << "_FREAK-Extractor"; break; }
		case 4: { descriptorExtractor = new cv::BriefDescriptorExtractor();	configurationTags << "_BRIEF-Extractor"; break; }		
		case 5: { descriptorExtractor = new cv::OrbDescriptorExtractor();	configurationTags << "_ORB-Extractor";  break; }
		case 6: { descriptorExtractor = new cv::BRISK();					configurationTags << "_BRISK-Extractor";  break; }
		
		default: break;
	}


	int bfNormType;
	Ptr<cv::flann::IndexParams> flannIndexParams/* = new cv::flann::AutotunedIndexParams()*/;
	if (descriptorExtractorSelection > 2) { // binary descriptors		
		bfNormType = cv::NORM_HAMMING;
		//flannIndexParams = new cv::flann::HierarchicalClusteringIndexParams();
		flannIndexParams = new cv::flann::LshIndexParams(12, 20, 2);
	} else { // float descriptors		
		bfNormType = cv::NORM_L2;
		flannIndexParams = new cv::flann::KDTreeIndexParams();
	}

	switch (descriptorMatcherSelection) {
		case 1: { descriptorMatcher = new cv::FlannBasedMatcher(flannIndexParams);	configurationTags << "_Flann-Matcher"; break; }
		case 2: { descriptorMatcher = new cv::BFMatcher(bfNormType, false);			configurationTags << "_BF-Matcher"; break; }
		default: break;
	}

	vector<string> imagesDBLevelOfDetail;

	switch (imagesDBLevelOfDetailSelection) {
		case 1: { imagesDBLevelOfDetail.push_back(REFERENCE_IMGAGES_DIRECTORY_VERY_LOW);	configurationTags << "_veryLowQualityImageDB"; break; }
		case 2: { imagesDBLevelOfDetail.push_back(REFERENCE_IMGAGES_DIRECTORY_LOW);			configurationTags << "_lowQualityImageDB"; break; }
		case 3: { imagesDBLevelOfDetail.push_back(REFERENCE_IMGAGES_DIRECTORY_MEDIUM);		configurationTags << "_mediumQualityImageDB"; break; }
		case 4: {
			imagesDBLevelOfDetail.push_back(REFERENCE_IMGAGES_DIRECTORY_VERY_LOW);
			imagesDBLevelOfDetail.push_back(REFERENCE_IMGAGES_DIRECTORY_LOW);
			imagesDBLevelOfDetail.push_back(REFERENCE_IMGAGES_DIRECTORY_MEDIUM);
			configurationTags << "_dynamicQualityImageDB"; break;
		}
		default: break;
	}

	switch (inliersSelectionMethod) {
		case 1: { inliersSelectionMethodFlagToUseGlobalMatch = true;	configurationTags << "_globalMatch"; break; }
		case 2: { inliersSelectionMethodFlagToUseGlobalMatch = false;	configurationTags << "_localMatch"; break; }
		default: break;
	}
	
	
	_imageDetector = new ImageDetector(featureDetector, descriptorExtractor, descriptorMatcher, _imagePreprocessor, configurationTags.str(), selectorTags.str(), imagesDBLevelOfDetail, inliersSelectionMethodFlagToUseGlobalMatch);
}


int CLI::selectImagesDBLevelOfDetail() {
	cout << "  => Select images database level of detail:\n";
	cout << "    1 - Very low  (256 pixels wide)\n";
	cout << "    2 - Low       (512 pixels wide)\n";
	cout << "    3 - Medium    (1024 pixels wide)\n";
	cout << "    4 - Dynamic   (Uses one of the previous 3 LOD according to the query image resolution)\n";

	return ConsoleInput::getInstance()->getIntCin("\n >>> Option [1, 4]: ", "Select one of the options above!", 1, 5);
}


int CLI::selectInliersSelectionMethod() {
	cout << "  => Select method to choose the best target match:\n";
	cout << "    1 - Best global match of feature points\n";
	cout << "    2 - Best local match of feature points (achieved considering the ROIs specified in image masks as standalone patches)\n";

	return ConsoleInput::getInstance()->getIntCin("\n >>> Option [1, 2]: ", "Select one of the options above!", 1, 3);
}


int CLI::selectFeatureDetector() {	
	cout << "  => Select feature detector:\n";
	cout << "    1 - SIFT\n";
	cout << "    2 - SURF\n";
	cout << "    3 - GFTT\n";
	cout << "    4 - FAST\n";	
	cout << "    5 - ORB\n";
	cout << "    6 - BRISK\n";
	cout << "    7 - STAR\n";
	cout << "    8 - MSER\n";

	return ConsoleInput::getInstance()->getIntCin("\n >>> Option [1, 7]: ", "Select one of the options above!", 1, 9);
}


int CLI::selectDescriptorExtractor() {	
	cout << "  => Select descriptor extractor:\n";
	cout << "    1 - SIFT\n";
	cout << "    2 - SURF\n";
	cout << "    3 - FREAK\n";
	cout << "    4 - BRIEF\n";			
	cout << "    5 - ORB\n";
	cout << "    6 - BRISK\n";	

	return ConsoleInput::getInstance()->getIntCin("\n >>> Option [1, 6]: ", "Select one of the options above!", 1, 7);
}


int CLI::selectDescriptorMatcher() {	
	cout << "  => Select descriptor matcher:\n";
	cout << "    1 - FlannBasedMatcher\n";
	cout << "    2 - BFMatcher\n";

	return ConsoleInput::getInstance()->getIntCin("\n >>> Option [1, 2]: ", "Select one of the options above!", 1, 3);
}


void CLI::showVersion() {
	cout << "+====================================================================================================+" << endl;
	cout << "|  Version 1.0 developed in 2013 for Augmented Reality course (5th year, 1st semester, MIEIC, FEUP)  |" << endl;
	cout << "|  Author: Carlos Miguel Correia da Costa (carlos.costa@fe.up.pt / carloscosta.cmcc@gmail.com)       |" << endl;
	cout << "+====================================================================================================+" << endl;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </Commnd Line user Interface>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
