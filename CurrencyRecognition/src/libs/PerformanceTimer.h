#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#ifdef _WIN32   // Windows system specific
	#include <windows.h>
#else           // Unix based system specific
	#include <sys/time.h>
#endif

#include <stdlib.h>
#include <string>
#include <sstream>

#include "TimeUtils.h"


// namespace specific imports to avoid namespace pollution
using std::string;
using std::stringstream;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <PerformanceTimer>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class PerformanceTimer {
	public:
		PerformanceTimer();
		virtual ~PerformanceTimer();

		void start();                             // start timer
		void stop();                              // stop the timer
		void reset();
		double getElapsedTimeInSec();             // get elapsed time in second
		double getElapsedTimeInMilliSec();        // get elapsed time in milli-second
		double getElapsedTimeInMicroSec();        // get elapsed time in micro-second
		string getElapsedTimeFormated();


	private:
		double elapsedTimeMicroSec;               // starting time in micro-second
		bool stopped;                             // stop flag

		#ifdef _WIN32
			LARGE_INTEGER frequencyWin;           // ticks per second
			LARGE_INTEGER startCountWin;
			LARGE_INTEGER endCountWin;
		#else
			timeval startCount;
			timeval endCount;
		#endif
		
		void calculateElapsedTimeMicroSec();
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </PerformanceTimer>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
