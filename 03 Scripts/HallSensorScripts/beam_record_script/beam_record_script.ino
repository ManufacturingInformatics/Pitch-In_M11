#include <SdFat.h>
#include <SPI.h>

// flag to force override generating filename
const bool fname_override=true;
// flag to enable printing vector to serial port
// used for debugging
const bool print_vals=true;

// LED blink delays to indicate different errors
const unsigned long SD_ERROR = 1000;
const unsigned long FILE_ERROR = 500;

// LED pin
const unsigned short LED_PIN = 11;
// current led state
bool ledState = false;

// variable to hold read result
long h0=0,h1=0,h2=0,h3=0,h4=0,h5=0;

/// Vector of variables
// array of pointers to the read variables
long* hArrayPt[] = {&h0,&h1,&h2,&h3,&h4,&h5};
// calculating size of array
unsigned int const hArraySize = sizeof(hArrayPt)/sizeof(*hArrayPt);
// calculating end address for looping
// pointer points to first element in array
//long** hArrayEnd = &hArrayPt[hArraySize-1];
// array of values that will hold the dereferenced values
long hArrayData[hArraySize] = {0};

//// print functions
/// print 1D vector
// print pointer array
template<size_t sz>
void printVector(long *(&arr)[sz]);
template<size_t sz>
void printVector(long (&arr)[sz]);

// sd card object and file object
SdFat sd;
SdFile myfile;

// timer variables used on led flashing and data timestamps
unsigned long currentmillis = millis();
unsigned long prevMillis = 0;
unsigned long startTime = 0;
// timer limit, in milliseconds
unsigned long timeLimit = 20000;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  // setup sensors
  pinMode(A0,INPUT);
  pinMode(A1,INPUT);
  pinMode(A2,INPUT);
  pinMode(A3,INPUT);
  pinMode(A4,INPUT);
  pinMode(A5,INPUT);
  // setup led
  pinMode(LED_PIN,OUTPUT);
  // ensure that the led is turned off
  digitalWrite(LED_PIN,LOW);
  ledState=false;
  
  if(!sd.begin(4,SPI_FULL_SPEED))
  {
    sd.initErrorHalt();
    // print error message
    Serial.println("initialization of SD card failed! Is it plugged in?");
    Serial.flush();
    // flash led to indicate an error
    while(true)
    {
      digitalWrite(LED_PIN,ledState);
      ledState!=ledState;
      delay(SD_ERROR);
    }
  }
  else
  {
    Serial.println("SD card ready!");
  }
  //printRootDirectory();
  // get new filename
  char* fname = getNewSDFilename();
  // if flag is set, override the generated filename
  // used for debugging and testing
  if(fname_override)
  {
    fname = (char*)"h0000000.csv";
  }
  // print filename
  Serial.println(fname);
  // open file
  if(myfile.open(fname,O_RDWR | O_CREAT | O_AT_END))
  {
    Serial.print(fname);
    Serial.println(" has been opened!");
  }
  else
  {
    // ensure that the file is closed
    myfile.close();
    // print error messages
    sd.errorHalt("opening file for write failed!");
    Serial.println("ERROR: Failed to open file!");
    // flash led to indicate an error
    while(true)
    {
      digitalWrite(LED_PIN,ledState);
      ledState!=ledState;
      delay(FILE_ERROR);
    }
  }
  // flush serial port to ensure that any status messages are printed
  Serial.flush();
  //exit(0);
}

void loop() {
  // record start time for reference in time stamp
  startTime = millis();
  // loop until time limit is reached
  while((millis()-startTime)<=timeLimit)
  {
    // read in and update variables
    h0 = analogRead(A0);
    h1 = analogRead(A1);
    h2 = analogRead(A2);
    h3 = analogRead(A3);
    h4 = analogRead(A4);
  
    /// update arrays+write to file
    /// printing the array is a debug feature
    // write time since start to file
    myfile.print(millis()-startTime);
    myfile.print(",");
    for(unsigned int i = 0;i<hArraySize;++i)
    {
      // dereference pointer array to update data array
      hArrayData[i] = *hArrayPt[i];
      // write data to file 
      myfile.print(*hArrayPt[i]);
      myfile.print(",");
    }
    myfile.println();
    if(print_vals)
    {
      printVector(hArrayData);
    }
    Serial.flush();
    delay(10);
  }
  // inform user that the time limit has been reached
  Serial.println("Stopping read! Time limit reached!");
  // close file
  myfile.close();
  Serial.flush();
  // exit and sit in inf loop
  exit(0);
}

// print pointer array
template<size_t sz>
void printVector(long *(&arr)[sz])
{
  for(size_t i=0;i<sizeof(arr)/sizeof(*arr);++i)
  {
    Serial.print(*arr[i]);
    Serial.print(" ");
  }
  Serial.println();
}

// print data array
template<size_t sz>
void printVector(long (&arr)[sz])
{
  for(size_t i=0;i<sz;++i)
  {
    Serial.print(arr[i]);
    Serial.print(" ");
  }
  //delay(1000);
  Serial.println();
}

void printRootDirectory()
{
  SdFile file;
  // open file object at the volume root directory and begin iterating thorugh
  while(file.openNext(sd.vwd(),O_READ))
  {
    file.printName(&Serial);
    Serial.print(' ');
    file.printFileSize(&Serial);
    Serial.println();
  }
  file.close();
}

char* getNewSDFilename()
{
  // buffer for filename to test
  static char fname[13]="";
  // counter for number in filename
  unsigned long ct = 0;
  // loop until filename is not found
  while(true)
  {
    // generate new filename to test
    sprintf(fname,"h%07d.csv",ct);
    // if the filename does not exist, break loop and return it to be used
    if(!sd.exists(fname))
    {
      return fname;
    }
    // increment counter
    ct++;
  }
}

void printDirectory(File dir, int numTabs)
{
  while(true)
  {
    // collect next file
    File entry = dir.openNextFile();
    // if no more files, break from loop
    if(!entry)
    {
      break;
    }
    // print the required number of tabs for the entry
    for(uint8_t i=0; i<numTabs;++i)
    {
      Serial.print('\t');
    }
    // print name of found object
    Serial.print(entry.name());
    // if object is a directory
    if(entry.isDirectory())
    {
      // add a slash and print contents
      Serial.println("/");
      printDirectory(entry,numTabs+1);
    }
    // if a file, print tabs and size
    else
    {
      Serial.print("\t\t");
      Serial.println(entry.size(),DEC);
    }
    // flush to make sure all the information is written
    Serial.flush();
    // close file
    entry.close();
  }
}
