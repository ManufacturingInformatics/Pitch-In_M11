#include <SdFat.h>
#include <SPI.h>

// flag to force override generating filename
const bool fname_override=true;

// variable to hold read result
long h0=0,h1=0,h2=0,h3=0,h4=0;

/// Vector of variables
// array of pointers to the read variables
long* hArrayPt[] = {&h0,&h1,&h2,&h3,&h4};
// calculating size of array
unsigned int const hArraySize = sizeof(hArrayPt)/sizeof(*hArrayPt);
// calculating end address for looping
// pointer points to first element in array
//long** hArrayEnd = &hArrayPt[hArraySize-1];
// array of values that will hold the dereferenced values
long hArrayData[hArraySize] = {0};

/// Grid of variables
// arranging the same pointers into a 2x2 array
long* hGrid[2][2] = {{&h0,&h1},{&h2,&h3}};
// store the size of the array
unsigned int const hGridRows = sizeof(hGrid)/sizeof(*hGrid);
unsigned int const hGridCols = sizeof(hGrid[0])/sizeof(*hGrid[0]);
// calculate end address for safe looping
//long** hGridEnd = &hGrid[hGridRows-1][hGridCols-1];
// grid of actual values that will be written
long hGridData[hGridRows][hGridCols] = {0};

//// print functions
/// print 1D vector
// print pointer array
template<size_t sz>
void printVector(long *(&arr)[sz]);
template<size_t sz>
void printVector(long (&arr)[sz]);

// print 2D grid of uniform size
// templates are used allow any sized 2d uniform array
template<size_t rows,size_t cols>
void printGrid(long *(&arr)[rows][cols]);
// data array
template<size_t rows,size_t cols>
void printGrid(long (&arr)[rows][cols]);

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
  pinMode(A0,INPUT);
  pinMode(A1,INPUT);
  pinMode(A2,INPUT);
  pinMode(A3,INPUT);
  pinMode(A4,INPUT);

  /*
  // Print array sizes to test that they've been created properly
  char gridSizeStr[20];
  sprintf(gridSizeStr,"Vector size: %d",hArraySize);
  Serial.println(gridSizeStr);
  sprintf(gridSizeStr,"Array size (rxc): %d x %d",hGridRows,hGridCols);
  Serial.println(gridSizeStr);

  sprintf(gridSizeStr,"Data vector size: %d",sizeof(hArrayData)/sizeof(*hArrayData));
  Serial.println(gridSizeStr);
  sprintf(gridSizeStr,"Data Array: %d x %d",sizeof(hGridData)/sizeof(*hGridData),sizeof(hGridData[0])/sizeof(*hGridData[0]));
  Serial.println(gridSizeStr);
  Serial.println();
  Serial.println("h0,h1,h2,h3,h4    ");
  */

  if(!sd.begin(4,SPI_FULL_SPEED))
  {
    sd.initErrorHalt();
    // print error message
    Serial.println("initialization of SD card failed! Is it plugged in?");
    Serial.flush();
    exit(1);
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
    fname = "h0000000.csv";
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
    myfile.close();
    sd.errorHalt("opening file for write failed!");
    Serial.println("ERROR: Failed to open file!");
    exit(2);
  }
  Serial.flush();
  //exit(0);
}

void loop() {
  startTime = millis();

  while((millis()-startTime)<timeLimit)
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
    printVector(hArrayData); 
    Serial.flush();
    delay(500);
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

template<size_t rows,size_t cols>
void printGrid(long *(&arr)[rows][cols])
{
  /// iterate over array using printVector to print rows
  // assuming 2D, uniform
  Serial.println("Grid pt");
  for(size_t r=0;r<rows;++r)
  {
    for(size_t c=0;c<cols;++c)
    {
      Serial.print(*arr[r][c]);
      Serial.print(" ");
    }
    Serial.println();
  }
  Serial.println();
}

template<size_t rows,size_t cols>
void printGrid(long (&arr)[rows][cols])
{
  /// iterate over array using printVector to print rows
  // assuming 2D, uniform 
  for(size_t r=0;r<rows;++r)
  {
    for(size_t c=0;c<cols;++c)
    {
      Serial.print(arr[r][c]);
      Serial.print(" ");
    }
    Serial.println();
  }
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
