#include <stdint.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <chrono>
#include <thread>
#include <math.h>
#include <stdlib.h>
#include <errno.h>
#include <iomanip>
#include <ctime>
#include <signal.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <linux/if_link.h>
#include <string.h>
#include <algorithm>
#include "/home/pi/mlx90640-library-master/headers/MLX90640_API.h"
#include "H5Cpp.h"

/*
 * rawval
 * ======
 *
 * David Miller, The University of Sheffield, 2019
 *
 * Collects frames of temperature  values from the camera at address 0x33
 * and writes the values to a HDF5 file.
 *
 * Arguments:
 * - FPS : Target frames per second
 * - Time : Time limit the program will run for in seconds.
 *
 * The HDF5 file is based off the current system date and time. The filename
 * is formatted as follows:
 *
 * pi-camera-data-IP-YYYY-MM-DDTHH-MM-SS.hdf5
 *
 * YYYY-MM-DDTHH-MM-SS is the date and time of creation and it based off
 * ISO format for date and time.
 *
 * IP is the ip of the Raspberry Pi with the dots removed. This part is to
 * ensure that data recorded on the same day by two different Pis have
 * different file names.
 *
 * The program is currently geared for ONLY one camera at address 0x33.
 * The information is saved into a resizeable dataset called pi-camera-1
 * which is increased by one frame per loop of collection.
 *
 * At the end the program, the following statistics are printed to screen.
 * - Total runtime
 * - Number of frames in each dataset stored in the file.
 * - Size of each dataset in Mb
 * - Estimated actual FPS based off the number of collected frames and
     elapsed time.
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

// flag for indicating if SIGINT is triggered
static volatile sig_atomic_t key_inter=0;

// function handler for is SIGINT is detected
void keyboard_interrupt(int)
{
    ++key_inter;
}

herr_t print_dset_info(hid_t loc_id, const char *name, const H5O_info_t *oinfo, void *opdata)
{
    // cast received data to unsigned long pointer
    unsigned long *frames = (unsigned long*)opdata;
    // open dataset using the location id
    hid_t obj = H5Oopen(loc_id,name,H5P_DEFAULT);
    // if the object is a dataset, print additional info
    // file only has datasets so anything else is root
    if(oinfo->type==H5O_TYPE_DATASET){
        std::cout << name << " : ";
        // open object as dataset
        H5::DataSet dset(obj);
        H5::DataSpace dspace = dset.getSpace();
        // get size of dataset
        hsize_t dims[3];
        int ndims = dspace.getSimpleExtentDims(dims,NULL);
        // print size
        std::cout << (unsigned long)(dims[0]) << "x"
              << (unsigned long)(dims[1]) << "x"
              << (unsigned long)(dims[2]) << " : "
              //get the size of the dataset in memory in Mb
              << (float)(dset.getInMemDataSize()/std::pow(1024.0,2.0)) << " Mb"
              << std::endl;
        // update and return number of frames
        *frames = (unsigned long)dims[2];
    }
    // close dataset
    H5Oclose(obj);
    return 0;
}

void printStats(H5::H5File f,std::chrono::seconds dur_secs, int target_fps)
{
    try{
    /// print statistics
    // print number of datasets
    std::cout << "Number of datasets: " << (unsigned long)f.getObjCount(H5F_OBJ_DATASET) << std::endl;
    // iterate over objects in the file performing print_dset_info on each one
    // num_frames is updated to the size of each dataset
    // as each dataset will be the same size, the end value will be
    // the number of frames collected
    unsigned long num_frames = 0;
    herr_t idx = H5Ovisit(f.getId(),H5_INDEX_NAME,H5_ITER_INC,print_dset_info,(void*)&num_frames);
    // print runtime
    std::cout << "Runtime: " << dur_secs.count() << " secs" << std::endl;
    // estimated FPS based off collected frames and elapsed time
    // performance measure
    std::cout << "Estimated " << (float)num_frames/(float)dur_secs.count() << " FPS"
              << " (" << target_fps << " FPS)" << std::endl;
    return;
    }
    catch(H5::Exception e)
    {
        std::cout << "HDF5 Exception!" << std::endl;
        return;
    }
}

int getRPiAddress(const char*& address)
{
    struct ifaddrs *ifaddr, *ifa;
    int family, s, n;
    char host[NI_MAXHOST];

    if (getifaddrs(&ifaddr) == -1) {
	perror("getifaddrs");
	exit(EXIT_FAILURE);
    }

/* Walk through linked list, maintaining head pointer so we
   can free list later */
    for (ifa = ifaddr, n = 0; ifa != NULL; ifa = ifa->ifa_next, n++) {
	if (ifa->ifa_addr == NULL)
	    continue;

	family = ifa->ifa_addr->sa_family;
        // the pi address for local networks is stored under wlan0
	if(strcmp(ifa->ifa_name,"wlan0")==0){
            // there are versions of wlan0, AF_INET and AF_INTET6
            // we want the AF_INET one
	    if (family == AF_INET) {
	         s = getnameinfo(ifa->ifa_addr,
		    (family == AF_INET) ? sizeof(struct sockaddr_in) :
		    sizeof(struct sockaddr_in6),
		    host, NI_MAXHOST,
		    NULL, 0, NI_NUMERICHOST);
		    // if failed to get name info, print error and return 1
		    if (s != 0) {
			std::cout << "getnameinfo() failed: %s\n" << gai_strerror(s) << std::endl;
			return 1;
		    }
                    // update address and break from loop
		    address = host;
		    break;
	    }
	}
    }

    // free structure as it is dynamically allocated
    freeifaddrs(ifaddr);
    return 0;
}

int main(int argc, char *argv[]){
    static uint16_t eeMLX90640[832];
    float emissivity = 0.8;
    uint16_t frame[834];
    static float mlx90640To[768];
    float eTa;
    static int fps = FPS;
    static int timelim = 0;
    static long frame_time_micros = FRAME_TIME_MICROS;
    char *p;
    // flag for indicating if an error occurred
    // used with try catch so the file can be closed afterwards
    bool errorFlag=false;
    // variables for getting
    // handle argument parsing for given frame rate
    if(argc > 1){
	// convert argument to long number
        fps = strtol(argv[1], &p, 0);
	// if error occured, print error message
        if (errno !=0 || *p != '\0') {
            std::cout << "Invalid framerate" << std::endl;
            return 1;
        }

        if(argc>2){
            timelim = strtol(argv[2],&p,0);
            if(errno!=0 || *p != '\0'){
                std::cout <<  "Invalid time limit. Reverting to no limit" << std::endl;
                timelim=0;
            }
            else
            {
                if(timelim<0){
                    std::cout << "Invalid time limit Time limit cannot be negative" << std::endl;
                    return 2;
                }
            }
        }
        frame_time_micros = 1000000/fps;
    }
    std::cout << "FPS: " << fps 
              << ", Time limit: " << ((timelim==0) ? "no limit":std::to_string(timelim)+" secs") 
              << std::endl;
    if(fps==64){
        frame_time_micros = 2000;
        std::cout << "Increasing time offset to " << frame_time_micros << std::endl;
    }
    // register signal handler for keyboard interrupt
    // on interrupt a global variable is decremented
    signal(SIGINT,keyboard_interrupt);

    /// get current local time as formatted string
    // get current time as a time_t object
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // create blank string of 45 chars in length initialized with null char
    std::string filename(50,'\0');
    // convert time to formatted string
    std::strftime(&filename[0],filename.size(),"-%Y-%m-%dT%H-%M-%S.hdf5",std::localtime(&now));
    const char* local_addr;
    // get address of the raspberry pi
    getRPiAddress(local_addr);
    // convert the address to a string
    std::string addr_str(local_addr);
    // remove dots from ip address
    addr_str.erase(std::remove(addr_str.begin(),addr_str.end(),'.'),addr_str.end());
    // append pi-camrea-data to address string
    addr_str.insert(0,"pi-camera-data-");
    // append modified address string to the time string to create filename
    filename.insert(0,addr_str);
    std::cout << "Creating file called " << filename << std::endl;

    // create file
    H5::H5File file(filename,H5F_ACC_TRUNC);
    // dataset sizes
    // new dimensions once extended, updated in loop
    hsize_t dimsext[3] = {24,32,1};
    // starter dimensions
    hsize_t dims[3] = {24,32,1};
    hsize_t maxdims[3] = {24,32,H5S_UNLIMITED};
    // size of chunks
    hsize_t chunk_dims[3] = {24,32,1};
    // offset for hyperslab
    hsize_t offset[3] = {0,0,0};
    // try catch for h5py operations
	try{
		// create dataspace for dataset
		H5::DataSpace dataspace(3,dims,maxdims);
		// dataspace for hyperslab
		H5::DataSpace slab;
		// modify dataset creation properties to enable chunking
		H5::DSetCreatPropList cparams;
		cparams.setChunk(3,chunk_dims);
		// set the initial value of the dataset
		float fill_val = 0.0;
		cparams.setFillValue(H5::PredType::NATIVE_FLOAT,&fill_val);

		// create dataset using set properties and dataspace
                std::cout << "Creating resizable dataset..." << std::endl;
		H5::DataSet dataset = file.createDataSet("pi-camera-1",H5::PredType::NATIVE_FLOAT,dataspace,cparams);
		// close properties list as  we don't need it anymore
                cparams.close();
		// log start time
		// used for delays between frames, time limit flag and statistics posted at the end
		auto frame_time = std::chrono::microseconds(frame_time_micros + OFFSET_MICROS);

                std::cout << "Initializing camera..." << std::endl;
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

		std::cout << "Entering loop..." << std::endl;
                // log start of the main loop
                auto start_prog = std::chrono::system_clock::now();
		while (1){
                        auto start = std::chrono::system_clock::now();
			MLX90640_GetFrameData(MLX_I2C_ADDR, frame);
			MLX90640_InterpolateOutliers(frame, eeMLX90640);

			eTa = MLX90640_GetTa(frame, &mlx90640); // Sensor ambient temprature
			MLX90640_CalculateTo(frame, &mlx90640, emissivity, eTa, mlx90640To); //calculate temprature of all pixels, base on emissivity of object 
			// select hyperslab
			slab = dataset.getSpace();
			slab.selectHyperslab(H5S_SELECT_SET,chunk_dims,offset);
			// write data to dataset
			dataset.write(mlx90640To,H5::PredType::NATIVE_FLOAT,dataspace,slab);
			// extend the dataset
			// set new size to increase number of frames by 1
			dimsext[2]+=1;
			dataset.extend(dimsext);

			/// Prep for next group
			// increase offset so the next frame is selected as a hyperslab
			offset[2]+=1;

			// update timer variables
			auto end = std::chrono::system_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			// get amount of time elapsed and cast to seconds
                        auto prog_elapsed = std::chrono::duration_cast<std::chrono::seconds>(end-start_prog);
                        // check keyboard interrupt flag
			// if not set, break from loop
			if(key_inter==1){
			    std::cout << "Keyboard Interrupt! Closing file" << std::endl;
			    printStats(file,prog_elapsed,fps);
                            file.close();
			    return -1;
			}

                        // checking if target elapsed time has passed
                        // timelim of 0 means no limit
                        if((timelim>0)&&(prog_elapsed.count()>timelim))
                        {
                            std::cout << "Time limit reached! Closing file" << std::endl;
                            printStats(file,prog_elapsed,fps);
                            file.close();
                            return 0;
                        }
                        // force thread to sleep to match fps
                        std::this_thread::sleep_for(std::chrono::microseconds(frame_time - elapsed));
		}
	}// end of try block
	// catch failure caused by the H5File operations
    catch(H5::FileIException error)
    {
        std::cout << "HDF5 File Exception! Closing file" << std::endl;
	// Close all objects and file.
	errorFlag=true;
	error.printErrorStack();
    }

    // catch failure caused by the DataSet operations
    catch(H5::DataSetIException error)
    {
	std::cout << "HDF5 Dataset Exception! Closing file" << std::endl;
        errorFlag=true;
	error.printErrorStack();
    }

    // catch failure caused by the DataSpace operations
    catch(H5::DataSpaceIException error)
    {
	std::cout << "HDF5 Dataspace Exception! Closing file" << std::endl;
        errorFlag=true;
	error.printErrorStack();
    }

    // close file
    file.close();
    // if the error flag has been set then return -1 to indicate an error
    // if no error or keyboard interrupt, return 0
    return (errorFlag)? -1:0;
}
