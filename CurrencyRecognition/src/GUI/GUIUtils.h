#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#define TEXT_MIN_SIZE 12
#define COLOR_TEXT Scalar(30, 255, 255)
#define COLOR_LABEL_BOX Scalar(45, 255, 255)
#define COLOR_LABEL_TEXT Scalar(214, 60, 5)

#define	WINDOW_HEADER_HEIGHT 32
#define WINDOW_FRAME_THICKNESS 8
#define WINDOW_OPTIONS_WIDTH 350
#define WINDOW_OPTIONS_HIGHT 935
#define WINDOW_DIGITS_HEIGHT 200
#define WINDOW_OPTIONS_TRACKBAR_HEIGHT 44
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </constants definitions> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <string>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


// project includes
#include "../Configs.h"


// namespace specific imports to avoid namespace pollution
using std::string;
using cv::Mat;
using cv::Rect;
using cv::Scalar;
using std::pair;
using cv::namedWindow;
using cv::moveWindow;
using cv::resizeWindow;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <GUIUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
namespace GUIUtils {
	/*!
	* \brief Draws a label in image in the top part of the signBoundingRect
	* \param text Text to draw
	* \param image Image where the text is going to be drawn
	* \param imageBoundingRect Rectangle with the region of interest were the text is going to be positioned inside the image
	*/
	void drawImageLabel(string text, Mat& image, const Rect& imageBoundingRect, float labelHeightPercentage = 0.05, float textThicknessPercentage = 0.05);


	void drawLabelInCenterOfROI(string text, Mat& image, const Rect& imageBoundingRect, float labelHeightPercentage = 0.25, float textThicknessPercentage = 0.05);


	/*!
	* \brief Adds a OpenCV HighGUI window, resized, aligned and positioned with the specified parameters
	* \param column Window column where the window will be moved
	* \param row Window row where the window will be moved
	* \param windowName Window name
	* \param numberColumns Number of window columns in which the screen is going to be divided
	* \param numberRows Number of window rows in which the screen is going to be divided		
	* \param imageWidth Override the computed image width
	* \param imageHeight Override the computed image height	
	* \param screenWidth Screen width
	* \param screenHeight Screen height
	* \param xOffset X offset to add to the original position
	* \param yOffset Y offset to add to the original position
	* \return Pair with he (x, y) position and the (width, height) of the window added
	*/
	pair< pair<int, int>, pair<int, int> > addHighGUIWindow(int column, int row, string windowName,
		int imageWidth = -1, int imageHeight = -1, int screenWidth = 1920, int screenHeight = 1080,
		int xOffset = 0, int yOffset = 0,
		int numberColumns = 3, int numberRows = 2);


	/*!
	* \brief Adds an OpenCV HighGUI track bar, resized, aligned and positioned with the specified parameters
	* \param windowName Window name
	* \param numberTrackBars Number of track bars that are going to be added to the window (to properly move the window that is going to be added)
	* \param cumulativeTrackBarPosition Number of track bars that are staked vertically in the windows above (to properly move the window that is going to be added)
	* \param trackBarWindowNumber Number of track bar windows that are going to be above (to properly move the window that is going to be added)
	* \param screenWidth Screen width	
	* \param xOffset Offset to adjust the x position
	* \param yOffset Offset to adjust the y position	
	* \return Pair with he (x, y) position and the (width, height) of the window added
	*/
	pair< pair<int, int>, pair<int, int> > addHighGUITrackBarWindow(string windowName, int numberTrackBars, int cumulativeTrackBarPosition, int trackBarWindowNumber,
		int screenWidth = 1920, 
		int xOffset = 0, int yOffset = 0);
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </GUIUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
