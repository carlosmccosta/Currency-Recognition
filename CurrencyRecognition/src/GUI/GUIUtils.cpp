#include "GUIUtils.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <GUIUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
void GUIUtils::drawImageLabel(string text, Mat& image, const Rect& signBoundingRect) {
	int textBoxHeight = (int)(signBoundingRect.height * 0.15);
	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	double scale = (double)textBoxHeight / 46.0;
	int thickness = (std::max)(1, (int)(textBoxHeight * 0.05));
	int baseline = 0;

	Rect textBoundingRect = signBoundingRect;
	textBoundingRect.height = (std::max)(textBoxHeight, TEXT_MIN_SIZE);
	//textBoundingRect.y -= textBoundingRect.height;

	cv::Size textSize = cv::getTextSize(text, fontface, scale, thickness, &baseline);
	cv::Point textBottomLeftPoint(textBoundingRect.x + (textBoundingRect.width - textSize.width) / 2, textBoundingRect.y + (textBoundingRect.height + textSize.height) / 2);

	cv::rectangle(image, signBoundingRect, COLOR_LABEL_BOX_HSV, 2);
	cv::rectangle(image, textBoundingRect, COLOR_LABEL_BOX_HSV, 2);
	cv::putText(image, text, textBottomLeftPoint, fontface, scale, COLOR_LABEL_TEXT_HSV, thickness);
}


pair< pair<int, int>, pair<int, int> > GUIUtils::addHighGUIWindow(int column, int row, string windowName,
	int imageWidth, int imageHeight, int screenWidth, int screenHeight,
	int xOffset, int yOffset,
	int numberColumns, int numberRows) {

	if (numberColumns < 1 || numberRows < 1)
		return pair< pair<int, int>, pair<int, int> >(pair<int, int>(0, 0), pair<int, int>(0, 0));

	int imageWidthFinal = imageWidth;
	if (imageWidthFinal < 10)
		imageWidthFinal = (screenWidth - WINDOW_OPTIONS_WIDTH) / 2;

	int imageHeightFinal = imageHeight;
	if (imageHeightFinal < 10)
		imageHeightFinal = (screenHeight - WINDOW_DIGITS_HEIGHT) / 2;


	int windowHeightFinal = ((screenHeight - WINDOW_DIGITS_HEIGHT) / numberRows);
	int windowWidthFinal = (imageWidthFinal * windowHeightFinal / imageHeightFinal);

	if ((windowWidthFinal * numberColumns + WINDOW_OPTIONS_WIDTH) > screenWidth) {
		windowWidthFinal = ((screenWidth - WINDOW_OPTIONS_WIDTH) / numberColumns);
		windowHeightFinal = imageHeightFinal * windowWidthFinal / imageWidthFinal;
	}

	namedWindow(windowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
	resizeWindow(windowName, windowWidthFinal - 2 * WINDOW_FRAME_THICKNESS, windowHeightFinal - WINDOW_FRAME_THICKNESS - WINDOW_HEADER_HEIGHT);

	int x = 0;
	if (column != 0) {
		x = windowWidthFinal * column;
	}

	int y = 0;
	if (row != 0) {
		y = windowHeightFinal * row;
	}

	x += xOffset;
	y += yOffset;

	moveWindow(windowName, x, y);

	return pair< pair<int, int>, pair<int, int> >(pair<int, int>(x, y), pair<int, int>(windowWidthFinal, windowHeightFinal));
}


pair< pair<int, int>, pair<int, int> > GUIUtils::addHighGUITrackBarWindow(string windowName, int numberTrackBars, int cumulativeTrackBarPosition, int trackBarWindowNumber,
	int screenWidth,
	int xOffset, int yOffset) {
	namedWindow(windowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

	int width = WINDOW_OPTIONS_WIDTH - WINDOW_FRAME_THICKNESS * 2;
	int height = numberTrackBars * WINDOW_OPTIONS_TRACKBAR_HEIGHT;
	resizeWindow(windowName, width, height);

	int x = (screenWidth - WINDOW_OPTIONS_WIDTH) + xOffset;
	int y = ((WINDOW_HEADER_HEIGHT + WINDOW_FRAME_THICKNESS) * trackBarWindowNumber + WINDOW_OPTIONS_TRACKBAR_HEIGHT * cumulativeTrackBarPosition) + yOffset;

	moveWindow(windowName, x, y);

	return pair< pair<int, int>, pair<int, int> >(pair<int, int>(x, y), pair<int, int>(width, height));
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </GUIUtils> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<