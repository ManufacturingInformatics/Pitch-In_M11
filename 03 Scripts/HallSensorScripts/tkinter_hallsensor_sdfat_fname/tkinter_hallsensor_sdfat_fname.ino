#include <TinkerKit.h>
#include <SdFat.h>

// LED Pin
#define LED_PIN 10

// LED flashing durations for different error occasions
#define FILE_ERROR 1000
#define SD_CARD_ERROR 100

// create hall sensor objects
TKHallSensor hs0(I0);
TKHallSensor hs1(I1);

// button to stop collecting data
TKButton stopButton(I2);

// LED state
int ledState = LOW;

// File object
SdFat sd;
SdFile myfile;
// timer variables used on led flashing and data timestamps
unsigned long currentmillis = millis();
unsigned long prevMillis = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  // setup status LED
  pinMode(LED_PIN,OUTPUT);
  // turn led off
  digitalWrite(LED_PIN,LOW);
  // initialize sd card
  // if it failed to initialize
  if(!sd.begin(4,SPI_FULL_SPEED))
  {
    sd.initErrorHalt();
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
  // create and open the file
  // check that the file has been opened
  if(myfile.open(fname,O_RDWR | O_CREAT | O_AT_END))
  {
    Serial.print(fname);
    Serial.println(" has been opened!");
  }
  else
  {
    sd.errorHalt("opening file for write failed!");
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
    // write values and number of milliseconds since start
    myfile.print(v1);
    myfile.print(',');
    myfile.print(v2);
    myfile.print(',');
    myfile.println(currentmillis);
    ////Print values
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
  printRootDirectory();
  // delay to ensure everything is printed
  Serial.flush();
  digitalWrite(LED_PIN,LOW);
  exit(0);
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
  static char fname[8];
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
