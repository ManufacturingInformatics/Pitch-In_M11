#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Windows.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <time.h>
#include <filesystem>
#include <signal.h>
#include <thread>
#include "KeyTimer.h"

// signal status, changed by signal handler function
volatile sig_atomic_t sigstatus = 0;

void signal_handler(int signal)
{
	sigstatus = signal;
}

// send a virtual press of F15 button 
void sendWakeupPress()
{
	// Input structure
	INPUT ip;
	// set as keyboard input
	ip.type = INPUT_KEYBOARD;
	ip.ki.time = 0;
	// set action as button press
	ip.ki.dwFlags = 0;
	ip.ki.wScan = 0;
	// set key to F15 button
	ip.ki.wVk = VK_F15;
	ip.ki.dwExtraInfo = 0;
	// send input
	//std::cout << "Sending input!" << std::endl;
	SendInput(1, &ip, sizeof(INPUT));
}

BOOL WINAPI CtrlHandler(DWORD ctrlType) 
{
	// while this function is being executed, the main thread is
	// still running, so setting the sigstatus var should close
	// the file
	switch (ctrlType)
	{
	// program is closed
	case CTRL_CLOSE_EVENT:
		std::cout << "Close event!" << std::endl;
		sigstatus = 2;
		// force thread to sleep to give opencv time to close file
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		return FALSE;
	// user logs off from the PC
	case CTRL_LOGOFF_EVENT:
		std::cout << "Logging off event!" << std::endl;
		sigstatus = 3;
		// force thread to sleep to give opencv time to close file
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		return FALSE;
	// PC is shutdown
	case CTRL_SHUTDOWN_EVENT:
		std::cout << "Shutting down event!" << std::endl;
		sigstatus = 4;
		// force thread to sleep to give opencv time to close file
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		return FALSE;
	
	default:
		sigstatus = 5;
		// force thread to sleep to give opencv time to close file
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		return FALSE;
	}
}


// CODE FROM https://stackoverflow.com/a/36696070
cv::Mat hwnd2mat(HWND hwnd)
{
	HDC hwindowDC, hwindowCompatibleDC;

	int height, width, srcheight, srcwidth;
	HBITMAP hbwindow;
	cv::Mat src;
	BITMAPINFOHEADER  bi;

	hwindowDC = GetDC(hwnd);
	hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
	SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

	RECT windowsize;    // get the height and width of the screen
	GetClientRect(hwnd, &windowsize);

	srcheight = windowsize.bottom;
	srcwidth = windowsize.right;
	height = windowsize.bottom / 1;  //change this to whatever size you want to resize to
	width = windowsize.right / 1;

	src.create(height, width, CV_8UC4);

	// create a bitmap
	hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
	bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
	bi.biWidth = width;
	bi.biHeight = -height;  //this is the line that makes it draw upside down or not
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 0;
	bi.biXPelsPerMeter = 0;
	bi.biYPelsPerMeter = 0;
	bi.biClrUsed = 0;
	bi.biClrImportant = 0;

	// use the previously created device context with the bitmap
	SelectObject(hwindowCompatibleDC, hbwindow);
	// copy from the window device context to the bitmap device context
	StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
	GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow

	// avoid memory leak
	DeleteObject(hbwindow);
	DeleteDC(hwindowCompatibleDC);
	ReleaseDC(hwnd, hwindowDC);

	return src;
}

int main(int argc, char** argv)
{
	// if the user has given a new destination path

	std::filesystem::path outpath;
	if (argc > 1)
	{
		outpath = argv[1];
	}
	else
	{
		// if no output path was given, set to current working directory
		outpath = std::filesystem::current_path();
	}
	// add signal handlers
	signal(SIGTERM, signal_handler);
	signal(SIGINT, signal_handler);
	signal(SIGABRT, signal_handler);
	// add control signal handler
	if (SetConsoleCtrlHandler(CtrlHandler, TRUE))
	{
		std::cout << "Control signal handler registered" << std::endl;
	}
	else
	{
		std::cout << "Failed to register control handler!" << std::endl;
		return 1;
	}

	/// construct filename
	// get current time as time_t object
	std::time_t currt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::time(&currt);
	struct tm timeinfo;
	localtime_s(&timeinfo, &currt);
	// create blank string
	std::string fullpath(50, '\0');
	// convert time to formatted string
	std::strftime(&fullpath[0], fullpath.size(), "-%Y-%m-%dT%H-%M-%S.mp4",&timeinfo);
	// append prefix
	fullpath.insert(0, "screencap");
	// append path to the start
	outpath /= fullpath;
	// create videowriter
	cv::String pathstr = outpath.string();
	// create empty writer
	cv::VideoWriter vwriter;
	// get handle for desktop window
	HWND hwndDesktop = GetDesktopWindow();
	// get handle for reading keyboard hits on console
	HANDLE hstdin = GetStdHandle(STD_INPUT_HANDLE);
	// image read
	cv::Mat src;
	// duration between key presses
	const int press_lim = 10;
	std::cout << "Setting delay between button pressed to " << press_lim << " secs" << std::endl;
	// create timed thread object to perform key presses
	KeyTimer keypress(press_lim);
	// start timed process
	keypress.setTimeout(sendWakeupPress);
	// loop until escape has been pressed
	// MSB is set if the escape key is pressed
	// also updated if pressed between calls
	double minVal;
	double maxVal;
	while(!GetAsyncKeyState(VK_ESCAPE))
	{
		// collect next frame 
		src = hwnd2mat(hwndDesktop);
		cv::minMaxLoc(src, &minVal, &maxVal);
		//std::cout << "Min val " << minVal << "Max val " << maxVal << std::endl;
		// if the video writer has not been opened/started
		if(!vwriter.isOpened())
		{
			// open video writer
			vwriter = cv::VideoWriter(outpath.string(), cv::VideoWriter::fourcc('M','P','4','A'), 30, cv::Size(src.cols,src.rows));
			// if it was successfully opened, inform user of location and image size
			if (vwriter.isOpened())
			{
				std::cout << "Opened writer at " << outpath << std::endl;
				std::cout << "Capturing " << src.cols << "x" << src.rows << std::endl;
			}
			// if attempts to open the writer failed, inform user citing target path
			else
			{
				std::cout << "Failed to open writer at " << outpath << std::endl;
			}
		}
		// if the writer has been opened, write the collected frame to the video
		if (vwriter.isOpened())
		{
			vwriter.write(src);
		}
		// if the global signal status variable has been set by the signal_handler function
		// in anyway, exit from loop
		// this is to ensure that the file is closed
		if (sigstatus > 0)
		{
			std::cout << "Signal received!" << std::endl;
			break;
		}
	}
	std::cout << "Exited! Releasing file" << std::endl;
	// clear flag to indicate to button thread to stop pressing button
	keypress.stop();
	std::cout << "Pressed the button " << keypress.getNumPresses() << " times" << std::endl;
	// release writer to secure file
	vwriter.release();
	return 0;
}