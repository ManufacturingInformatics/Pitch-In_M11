#include <TinkerKit.h>
#include <SPI.h>
#include <SD.h>
// LED Pin
#define LED_PIN 10

// LED flashing durations for different error occasions
#define FILE_ERROR = 1000
#define SD_CARD_ERROR = 100

// create hall sensor objects
TKHallSensor hs0(I0);
TKHallSensor hs1(I1);

// button to stop collecting data
TKButton stopButton(I2);

// LED state
int ledState = LOW;

// File object
File myfile;

unsigned long currentmillis = millis();
unsigned long prevMillis = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  // setup status LED
  pinMode(LED_PIN,OUTPUT);
  // turn led off
  digitalWrite(LED_PIN,LOW);
  // check state of sd card
  // if it failed to initialize
  if(!SD.begin(4))
  {
    // print error message
    Serial.println("initialization of SD card failed! Is it plugged in?");
    Serial.flush();
    // flash led at set rate to indicate error
    while(true)
    {
      currentmillis = millis();
      if(currentmillis-prevMillis>=SD_CARD_ERROR)
      {
        digitalWrite(LED_PIN,ledState);
        ledState =!ledState;
        prevMillis=currentmillis;
      }
    }
  }
  else
  {
    //Serial.println("SD card ready!");
  }
  
  // get and print name of new file
  char* fname = getNewSDFilename();
  //Serial.print("New filename ");
  //Serial.println(fname);
  // create and open the file
  myfile = SD.open(fname,FILE_WRITE);
  // check that the file has been opened
  if(myfile)
  {
    Serial.println("File has been opened!");
  }
  else
  {
    Serial.println("ERROR: Failed to open file!");
    // if the file failed to open, sit and flash led
    while(true)
    {
      currentmillis = millis();
      if(currentmillis-prevMillis>=FILE_ERROR)
      {
        digitalWrite(LED_PIN,ledState);
        ledState =!ledState;
        prevMillis=currentmillis;
      }
    }
  }
}

void loop() {
  // variables
  long v1;
  long v2;
  Serial.println("Starting read");
  // turn led on to indicate no problems
  digitalWrite(LED_PIN,HIGH);
  // log start time
  long startt = millis();
  // loop until button press, press state is toggled
  while(stopButton.readSwitch() == LOW)
  {
    // Read timestamp and sensor values
    currentmillis = millis()-startt;
    v1 = hs0.read();
    v2 = hs1.read();
    // if file is OK to write to, write values
    if(myfile)
    {
      // write values and number of milliseconds since start
      myfile.print(v1);
      myfile.print(',');
      myfile.print(v2);
      myfile.print(',');
      myfile.println(currentmillis);
    }
    //Serial.print(v1-505);
    //Serial.print("\t");
    //Serial.println(v2-505);
    // delay of 10 ms between reads
    delay(10);
  }
  Serial.println("Stopping read");
  // turn off led to indicate that logging has finished
  // closing file
  myfile.close();
  // print contents of SD card to show a file has been written
  //Serial.println("Printing contents of SD card");
  File root = SD.open("/");
  printDirectory(root,0);
  root.close();
  // delay to ensure everything is printed
  Serial.flush();
  digitalWrite(LED_PIN,LOW);
  exit(0);
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

char* getNewSDFilename()
{
  // buffer for filename to test
  static char fname[8];
  // counter for number in filename
  unsigned long ct = 0;
  // loop until filename is not found
  while(true)
  {
    // generate new filename to test
    sprintf(fname,"h%07d.csv",ct);
    // if the filename does not exist, break loop and return it to be used
    if(!SD.exists(fname))
    {
      break;
    }
    // increment counter
    ct++;
  }
  return fname;
}
