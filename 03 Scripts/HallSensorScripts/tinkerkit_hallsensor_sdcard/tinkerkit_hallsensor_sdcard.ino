#include <TinkerKit.h>
#include <SPI.h>
#include <SD.h>

#define LED_ON 1023
#define LED_OFF 0

// create hall sensor objects
TKHallSensor hs0(I0);
TKHallSensor hs1(I1);

// button to stop collecting data
TKButton stopButton(I2);

// LED to indicate status
TKLed statusLED(O0);

// File object
File myfile;
// variables
long v1;
long v2;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  if(!SD.begin(4))
  {
    Serial.println("initialization of SD card failed! Is it plugged in?");
    while(1)
    {
      statusLED.on();
      delay(2000);
      statusLED.off();
    }
  }
  // create and open the file
  myfile = SD.open("test2.csv",FILE_WRITE);
  if(myfile)
  {
    Serial.println("test2.csv file opened");
  }
  statusLED.on();
}

void loop() {
  Serial.println("Starting read");
  // loop until button press
  while(stopButton.readSwitch() == LOW)
  {
    // Read sensor values and print it
    v1 = hs0.read();
    v2 = hs1.read();
    // if file is OK to write to, wriite valyes
    if(myfile)
    {
      // write to file
      myfile.print(v1);
      myfile.print(',');
      myfile.print(v2);
      myfile.println();
    }
    Serial.print(v1-505);
    Serial.print("\t");
    Serial.print(v2-505);
    Serial.println();
    // delay of 10 ms between reads
    delay(10);
  }
  Serial.println("Stopping read");
  // closing file
  myfile.close();
  // print contents of SD card to show a file has been written
  Serial.println("Printing contents of SD card");
  File root = SD.open("/");
  printDirectory(root,0);
  root.close();
  Serial.println("Turning LED Off");
  statusLED.off();
  // delay to ensure everything is printed
  delay(200);
  exit(0);
}

void printDirectory(File dir, int numTabs)
{
  char data[20];
  while(true)
  {
    File entry = dir.openNextFile();
    // if no more file, break from loop
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
    // close file
    entry.close();
  }
}
