import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

hall_pin = 0
# set specific pin as input
# set pull up resistor as up 
GPIO.setup(hall_pin,GPIO.IN,pull_up_down = GPIO.PUD_UP)

try:
    while True:
        # read sensor
        hall_v = GPIO.input(hall_pin)
        # print value
        print(hall_v)
        # sleep for 1 sec
        time.sleep(1)
# break on keyboard interrupt
except KeyboardInterrupt:
    GPIO.cleanup()

