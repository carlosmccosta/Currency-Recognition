#include "TimeUtils.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <TimeUtils>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
string TimeUtils::formatSecondsToDate(double seconds) {
	double secondsFinal = fmod(seconds, 60);

	double minutes = seconds / 60.0;
	unsigned short minutesFinal = (unsigned short) fmod(minutes, 60.0);

	double hours = minutes / 60;
	unsigned short hoursFinal = (unsigned short) fmod(hours, 24.0);

	unsigned short daysFinal = (unsigned short) (hours / 24.0);

	stringstream timeFormated;
	if (daysFinal != 0) {
		timeFormated << daysFinal << "d";
	}

	if (hoursFinal != 0 || daysFinal != 0) {
		if (hoursFinal < 10) {
			timeFormated << 0;
		}
		timeFormated << hoursFinal << "h";
	}

	if (minutesFinal != 0 || hoursFinal != 0 || daysFinal != 0) {
		if (minutesFinal < 10) {
			timeFormated << 0;
		}
		timeFormated << minutesFinal << "m";
	}

	if ((minutesFinal != 0 || hoursFinal != 0 || daysFinal != 0) && secondsFinal < 10 &&  secondsFinal >= 1) {
		timeFormated << 0;
	}
	timeFormated << secondsFinal << "s";

	return timeFormated.str();
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </TimeUtils>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
