import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

hall_pin = 0
# set specific pin as input
# set pull up resistor as up 
GPIO.setup(hall_pin,GPIO.IN,pull_up_down = GPIO.PUD_UP)

def print_change():
    print(GPIO.input(hall_pin))
# register the hall sensor pin to detect rising and falling edge 
GPIO.add_event_detect(hall_pin,GPIO.BOTH)
# register the function print_change to occur whenever a registered event
# is detected
GPIO.add_event_callback(hall_pin,print_change)