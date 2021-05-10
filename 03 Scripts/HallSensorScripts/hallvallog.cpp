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
#include <netinet/if_ether.h>
#include <ifaddrs.h>
#include <linux/if_link.h>
#include <string.h>
#include <algorithm>
// reference to wiringPi
// http://wiringpi.com/
#include <wiringPi.h>
#include <mcp3004.h>
#include "H5Cpp.h"

// pin base for the chip
// inputs are connected to pins 100 through 107 on the R Pi
#define BASE 100
// spi channel, 0 or 1
// peripheral it is connected to
#define SPI_CHAN 0

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
        hsize_t dims[2];
        int ndims = dspace.getSimpleExtentDims(dims,NULL);
        // print size
        std::cout << (unsigned long)(dims[0]) << "x"
              << (unsigned long)(dims[1])
              //get the size of the dataset in memory in Mb
              << (float)(dset.getInMemDataSize()/std::pow(1024.0,2.0)) << " Mb"
              << std::endl;
        // update and return number of frames
        *frames = (unsigned long)dims[0];
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


void getRPiMACAddress(const char*& address)
{
    struct ifaddrs *ifaddr, *ifa;
    int family, s, n;
    char host[NI_MAXHOST];

    if (getifaddrs(&ifaddr) == -1) {
		return;
    }
	
	// if ifaddr structure was not initialized, exit early
	if(!ifaddr)
	{
		return;
	}

	int32_t sd = socket(PF_INET,SOCK_DGRAM,0);
	// attempt to establish socket, if it fails or error ?
	if(sd<0)
	{
		// free interface addresses structure and return
		freeifaddrs(ifaddr);
		return;
	}
/* Walk through linked list, maintaining head pointer so we
   can free list later */
    for (ifa = ifaddr, n = 0; ifa != NULL; ifa = ifa->ifa_next, n++) 
	{
		if (ifa->ifa_addr == NULL)
			continue;

		family = ifa->ifa_addr->sa_family;
        // the pi address for local networks is stored under wlan0
		if(strcmp(ifa->ifa_name,"wlan0")==0)
		{
			if(ifa->ifa_data != 0)
			{
				struct ifreq req;
				// copy the device name into string
				strcpy(req.ifr_name,ifa->ifa_name);
				// attempt to access underlying kernel parameters associated with
				// the device associated with req
				if( ioctl( sd, SIOCGIFHWADDR, &req ) != -1 )
				{
					// get mac address, arrayn of uint8
					uint8_t* mac = (uint8_t*)req.ifr_ifru.ifru_hwaddr.sa_data;
					// convert to string and update address pointer
					asprintf(address,"%s",ether_ntoa((struct ether_addr*)mac));
				}
			}
		}
    }

    // free structure as it is dynamically allocated
    freeifaddrs(ifaddr);
    return;
}

float convertVoltToGauss(V0,V,S)
{
	return (V0-V)/S
}

int main(int argc, char *argv[]){
	// sensitivity of the sensor in mV/G
	const float sensitivity = 1.3;
	// voltage when there are no magnets arouund, mV
	const float V0 = 2.5;
	// measured voltage
	float V = 0.0;
	// estimated field strength
	float H = 0.0;
	// analog channel to read from chip
	const int chan = 0;
	// data-time element to write to file
	float data[2];
	// mac address of wlan0
	const char* mac_addr;
	// GPIO pin number of the hall sensor input_iterator
	const int hall_pin = 0;
	
	// set address as fi
	// register keyboard interrupt signal handler
	signal(SIGINT,keyboard_interrupt);
	
	// setup wiring pi library
	wiringPi();
	// setup mcp3008 chip for reading
	// same function as mcp3004
	mcp3004Setup(BASE,SPI_CHAN);
	
	// get system date and time
	std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    // create blank string of 45 chars in length initialized with null char
    std::string filename(50,'\0');
    // convert time to formatted string
    std::strftime(&filename[0],filename.size(),"-%Y-%m-%dT%H-%M-%S.hdf5",std::localtime(&now));
	
	// get wlan0 mac address to use in filename creation
	getRPiMACAddress(mac_addr);
	// convert mac address to string
	std::string mac_str(mac_addr);
	
	// remove semicolons from ip address
    mac_str.erase(std::remove(mac_str.begin(),mac_str.end(),':'),mac_str.end());
    // append pi-camrea-data to address string
    mac_str.insert(0,"hall-data-");
    // append modified address string to the time string to create filename
    filename.insert(0,mac_str);
    std::cout << "Creating file called " << filename << std::endl;
	
	// create file
    H5::H5File file(filename,H5F_ACC_TRUNC);
    // dataset sizes
    // new dimensions once extended, updated in loop
    hsize_t dimsext[2] = {1,2};
    // starter dimensions
    hsize_t dims[2] = {1,2};
    hsize_t maxdims[2] = {H5S_UNLIMITED,2};
    // size of chunks
    hsize_t chunk_dims[2] = {1,2};
    // offset for hyperslab
    hsize_t offset[2] = {0,0};
	
	try{
		// create dataspace for dataset
		H5::DataSpace volt_dspace(2,dims,maxdims);
		H5::DataSpace gauss_dspace(2,dims,maxdims);
		// dataspace for hyperslab
		H5::DataSpace slab;
		// modify dataset creation properties to enable chunking
		H5::DSetCreatPropList cparams;
		cparams.setChunk(2,chunk_dims);
		// set the initial value of the dataset
		float fill_val = 0.0;
		cparams.setFillValue(H5::PredType::NATIVE_FLOAT,&fill_val);

		// create dataset using set properties and dataspace
        std::cout << "Creating resizable dataset..." << std::endl;
		H5::DataSet voltset = file.createDataSet("voltage",H5::PredType::NATIVE_FLOAT,volt_dspace,cparams);
		H5::DataSet magset  = file.createDataSet("gauss",H5::PredType::NATIVE_FLOAT,gauss_dspace,cparams); 
		// close properties list as  we don't need it anymore
        cparams.close();
		
		// log start of program
		auto start_prog = std::chrono::system_clock::now();
		while(1)
		{
			/* INSERT READ VOLTAGE FUNCTION HERE */
			// read voltage from channel, chan
			V = analogRead(BASE+chan);
			
			// package voltage as dataset
			data = {V,std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_prog)};
			// get dataspace
			slab = volt_dspace.getSpace();
			slab.selectHyperslab(H5S_SELECT_SET,chunk_dims,offset);
			// write data to dataset
			voltset.write(data,H5::PredType::NATIVE_FLOAT,volt_dspace,slab);
			
			// convert voltage to Gauss
			data[0] = convertVoltToGauss(V0,V,sensitivity);
			// get slice of data to write to
			slab = gauss_dspace.getSpace();
			slab.selectHyperslab(H5S_SELECT_SET,chunk_dims,offset);
			// write estimate magnetic strength to file
			magset.write(data,H5::PredType::NATIVE_FLOAT,gauss_dspace,slab);
			
			// extend the datasets
			// set new size to increase number of rows by 1
			dimsext[0]+=1;
			volt_dspace.extend(dimsext);
			gauss_dspace.extend(dimsext);
			/// Prep for next group
			// increase offset so the next frame is selected as a hyperslab
			offset[0]+=1;
		}
	}
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