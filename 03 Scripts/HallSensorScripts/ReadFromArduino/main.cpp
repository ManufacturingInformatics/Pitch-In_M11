#include "SerialClass.h"
#include <iostream>

int main(int argc, char* argv[])
{
	// create serial port manager
	Serial* SP = new Serial("\\\\.\\COM4");

	// if successfully connected, inform user of port name
	if (SP->IsConnected())
	{
		std::cout << "Connected to " << SP->GetPortString() << std::endl;
	}

	// setup read buffer
	char rin[20] = "";
	char schar[1] = "";
	int dataLength = 19;
	int readResult = 0;
	// variables for all read
	char* superBuffer;
	size_t allReadLength = 0;
	while (SP->IsConnected())
	{
		/*
		// read data into buffer
		// returns the number of bytes read
		readResult = SP->ReadData(rin, dataLength);
		// terminate content with null character
		rin[readResult] = 0;
		// print contents
		std::cout << rin << std::endl;
		std::cout << "count " << readResult << std::endl;*/

		for (int i = 0; i < 17; ++i)
		{
			// read in single character
			SP->ReadData(schar, 1);
			// update array
			rin[i] = *schar;
		}
		std::cout << rin << std::endl;

		// read all data in buffer
		//superBuffer = SP->ReadAllData(allReadLength);
		//std::cout << superBuffer << std::endl;
		// press character to get next batch
		Sleep(300);
		std::getchar();
	}
	return 0;
}