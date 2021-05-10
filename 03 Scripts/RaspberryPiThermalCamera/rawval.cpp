#include <stdint.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include <math.h>
#include <stdlib.h>
#include <errno.h>
#include "headers/MLX90640_API.h"

/*
 * rawval
 * ======
 * outputs 24x32 temperature values to stdout
 *
 * David Miller, 2019, The University of Sheffield
 * 
 * Based off the rawrgb example supplied in the library.
 * Sends 16-bit temperature values along stdout so it can be picked up by
 * other programs such as logto-hdfd5.py
 * 
 * Arguments:
 *  - fps : Target FPS for collection. Default: 4 fps
*/


#define MLX_I2C_ADDR 0x33

#define IMAGE_SCALE 5

// Valid frame rates are 1, 2, 4, 8, 16, 32 and 64
// The i2c baudrate is set to 1mhz to support these
#define FPS 16
#define FRAME_TIME_MICROS (1000000/FPS)

// Despite the framerate being ostensibly FPS hz
// The frame is often not ready in time
// This offset is added to the FRAME_TIME_MICROS
// to account for this.
#define OFFSET_MICROS 850
// size of floats in c++
#define FLOAT_SIZE 4
// number of bytes in the frame
#define IMAGE_SIZE 768*FLOAT_SIZE

int main(int argc, char *argv[]){
    static uint16_t eeMLX90640[832];
    float emissivity = 0.8;
    uint16_t frame[834];
    static float mlx90640To[768];
    float eTa;
    static int fps = FPS;
    static long frame_time_micros = FRAME_TIME_MICROS;
    char *p;
   
    if(argc ==2){
        std::cout << "Converting fps" << std::endl;
	// convert argument to long number
        fps = strtol(argv[1], &p, 0);
	// if error occured, print error message
        if (errno !=0 || *p != '\0') {
            fprintf(stderr, "Invalid framerate\n");
            return 1;
        }
        frame_time_micros = 1000000/fps;
    }
    auto frame_time = std::chrono::microseconds(frame_time_micros + OFFSET_MICROS);

    MLX90640_SetDeviceMode(MLX_I2C_ADDR, 0);
    MLX90640_SetSubPageRepeat(MLX_I2C_ADDR, 0);
    switch(fps){
        case 1:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b001);
            break;
        case 2:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b010);
            break;
        case 4:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b011);
            break;
        case 8:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b100);
            break;
        case 16:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b101);
            break;
        case 32:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b110);
            break;
        case 64:
            MLX90640_SetRefreshRate(MLX_I2C_ADDR, 0b111);
            break;
        default:
            fprintf(stderr, "Unsupported framerate: %d\n", fps);
            return 1;
    }
    MLX90640_SetChessMode(MLX_I2C_ADDR);

    paramsMLX90640 mlx90640;
    MLX90640_DumpEE(MLX_I2C_ADDR, eeMLX90640);
    MLX90640_SetResolution(MLX_I2C_ADDR, 0x03);
    MLX90640_ExtractParameters(eeMLX90640, &mlx90640);

    std::cout << "FPS " << fps << " emissivity " << emissivity << std::endl;

    while (1){
        auto start = std::chrono::system_clock::now();
        MLX90640_GetFrameData(MLX_I2C_ADDR, frame);
        MLX90640_InterpolateOutliers(frame, eeMLX90640);

        eTa = MLX90640_GetTa(frame, &mlx90640); // Sensor ambient temprature
        MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To); //calculate temprature of all pixels, base on emissivity of object 

	MLX90640_BadPixelsCorrection((&mlx90640)->brokenPixels,mlx90640To,1,&mlx90640);
	MLX90640_BadPixelsCorrection((&mlx90640)->outlierPixels,mlx90640To,1,&mlx90640);


	// write temperature values to stdout
	fwrite(&mlx90640To,sizeof(float),768,stdout);

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::this_thread::sleep_for(std::chrono::microseconds(frame_time - elapsed));
    }

    return 0;
}

