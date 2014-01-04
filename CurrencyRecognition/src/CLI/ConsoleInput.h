#pragma once


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// std includes
#include <iostream>
#include <string>
#include <sstream>
#include <limits>

// namespace specific imports to avoid namespace pollution
using std::string;
using std::stringstream;
using std::getline;
using std::cin;
using std::cout;
using std::endl;
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <includes> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Console input library>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class ConsoleInput {
	public:
		static ConsoleInput* getInstance();

		void flushStandardInput();
		void getUserInput();
		string getLineCin();
		void clearConsoleScreen();
		int getIntCin(const char* message, const char* errorMessage, int min = 0, int size = std::numeric_limits<int>::max());

		template<typename NumberType>
		NumberType getNumberCin(const char* message, const char* errorMessage, NumberType min = 0, NumberType size = std::numeric_limits<NumberType>::max()) {
			NumberType number;
			do {
				cout << message << std::flush;

				string numberStr = getLineCin();
				stringstream strstream(numberStr);

				if (strstream >> number) {
					if (number >= min && number < size)
						break;
					else
						cout << errorMessage << endl;
				} else {
					cout << errorMessage << endl;
				}

			} while (true);

			return number;
		}

		bool getYesNoCin(const char* message, const char* errorMessage = "    -> Incorrect answer! Insert Y or N!\n");


	private:
		ConsoleInput() {}
		virtual ~ConsoleInput() {}
		static ConsoleInput* instance;
};
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </Console input library>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
