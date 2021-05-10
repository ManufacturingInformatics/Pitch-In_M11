//#include <TinkerKit.h>

//#define LED_ON A023
//#define LED_OFF 0

// create hall sensor objects
//TKHallSensor hs0(I0);
//TKHallSensor hs1(I1);

// button to stop collecting data
//TKButton stopButton(I2);

// LED to indicate status
//TKLed //statusLED(O0);

long h0=0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(A0,INPUT);
}

void loop() {
  //statusLED.on();
 // Serial.println("Starting read");
  // Read sensor values and print it
  h0 = analogRead(A0);
  Serial.println(h0);
  // delay of A0 ms between reads
  delay(10);
}
