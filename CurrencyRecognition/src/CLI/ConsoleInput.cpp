#include "ConsoleInput.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Console input library>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
void ConsoleInput::getUserInput() {
	cout << "Press ENTER to continue..." << endl;
	string temp;
	getline(cin, temp);
}


string ConsoleInput::getLineCin() {
	string input;
	getline(cin, input);
	
//	cin.clear();
//	cin.sync();
	return input;
}


void ConsoleInput::clearConsoleScreen() {
	for (size_t i = 0; i < 80; ++i) {
		cout << "\n";
	}
	
	cout << endl;
}


int ConsoleInput::getIntCin(const char* message, const char* errorMessage, int min, int size) {
	int number;
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


bool ConsoleInput::getYesNoCin(const char* message, const char* errorMessage) {
	bool stop = false;
	bool incorrectOption;
	string option;
	
	do {
		cout << message << std::flush;
		
		option = getLineCin();
		if ((option == "Y") || (option == "y")) {
			stop = true;
			incorrectOption = false;
		} else if ((option == "N") || (option == "n")) {
			stop = false;
			incorrectOption = false;
		} else {
			cout << errorMessage << endl;
			incorrectOption = true;
		}
	} while (incorrectOption);
	
	return stop;
}


ConsoleInput* ConsoleInput::getInstance() {
	if (instance == NULL) {
		instance = new ConsoleInput();
	}
	
	return instance;
}

ConsoleInput* ConsoleInput::instance = NULL;


void ConsoleInput::flushStandardInput() {
	cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
//	cin.clear();
//	cin.sync();
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </Console input library>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
