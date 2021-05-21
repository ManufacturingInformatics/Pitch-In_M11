import json
import math
import pickle
import time

# import context  # Ensures paho is in PYTHONPATH
import paho.mqtt.publish as publish
import RPi.GPIO as GPIO  # import GPIO

from hx711 import HX711  # import the class HX711


def load_setup():
    newset = int(
        input('Q-1: New setup? 1/0: (if no, will use the setup changed last time): '))
    if newset == 1:
        res = reset()
        hx, ratio = reset_loadcell(res)
        res["tags"]["sensor_ratio"] = ratio
        with open('setup.pickle', 'wb') as f:
            pickle.dump(res, f)
        print("config file saved\n")
    else:
        with open('setup.pickle', 'rb') as f:
            res = pickle.load(f)
        print("config file loaded\n")
        hx, ratio = reset_loadcell(res)
    return hx, res


def reset():
    # ====================================================
    # %% 1. What is this device

    # get hostname
    import socket
    hostname = socket.gethostname()

    print("Your Computer Name is:" + hostname)

    # get mac address

    def getMAC(interface='eth0'):
        # Return the MAC address of the specified interface
        try:
            str = open('/sys/class/net/%s/address' % interface).read()
        except:
            str = "00:00:00:00:00:00"
        return str[0:17]
    maceth = getMAC('eth0')
    macwlan = getMAC('wlan0')
    macadd = (macwlan, maceth,)
    print(macadd)

    # ====================================================
    # %% 2. Who should this device talk to

    # get MQTT address
    global MQTT_CHECK
    global MQTT_SERVER
    global MQTT_TOPIC

    MQTT_CHECK = int(input(
        'Q-2: MQTT active? if not the data would not be sent to server and will not be saved. 1/0:  '))

    # MQTT Settings
    if MQTT_CHECK:
        MQTT_SERVER = input('MQTT host ip:')
    else:
        MQTT_SERVER = "localhost"

    # setup MQTT topic
    MQTT_TOPIC = "boyang/iot/loadcell/" + \
        hostname

    # ====================================================
    # %% 3. How did this device connect with sensor(s)? GPIO

    # GPIO setup
    # VCC to PIN 1 (5V)
    # GND to PIN 6
    print("Hit: sensor #0: pin_DATA = 21, pin_CLOCK = 20; sensor #1: pin_DATA = 26, pin_CLOCK = 19")
    pin_DATA = int(input("Q-3: What is the DATA pin "))
    # 19  # GPIO 20 = PIN 38 CLOCK CLK
    pin_CLOCK = int(input('Q-4: What is the CLOCK pin '))
    # config of the sensor (max scale)
    sensor_spec = int(input('Q-5: what is the KG range for this loadcell: '))
    if not sensor_spec:
        sensor_spec = 0
        print("default is 0")

    sensor = {
        'pin_DATA': pin_DATA,
        'pin_CLOCK': pin_CLOCK,
        'sensor_spec': sensor_spec
    }

    feq = int(input('Q-6: how many data points average one reading?: '))
    fr_filter = float(
        input('Q-7: Filter-what is the max flow rate (g/sec)?: '))
    res = {
        "measurement": "loadcell",
        "tags": {
            "hostname": hostname,
            "mac_eth0": maceth,
            "mac_wlan0": macwlan,

            "MQTT_CHECK": MQTT_CHECK,
            "MQTT_SERVER": MQTT_SERVER,
            "MQTT_TOPIC": MQTT_TOPIC,

            "sensors": sensor,
            "sensor_type": "loadcell",
            "sensor_ratio": None,

            "feq": feq,
            "filter": fr_filter
        },
        "fields": {
            "f_raw": None,  # signals from the loadcell
            "f_nominal": None,  # convert the horizontal weight to vertical weight, with the formula
        }
    }

    return res


def reset_loadcell(res):
    # %% 1. Clear GPIO

    GPIO.setmode(GPIO.BCM)  # set GPIO pin mode to BCM numbering

    # %% 2. set sensor objective
    # Create an object hx which represents your real hx711 chip
    # Required input parameters are only 'dout_pin' and 'pd_sck_pin'
    # hx = HX711(dout_pin=21, pd_sck_pin=20)
    hx = HX711(dout_pin=res['tags']['sensors']['pin_DATA'],
               pd_sck_pin=res['tags']['sensors']['pin_CLOCK'])
    # measure tare and save the value as offset for current channel
    # and gain selected. That means channel A and gain 128

    # %% 3. Calibrate each sensor

    input('Please clear the rack, and press Enter')

    # Before we start, reset the hx711 ( not necessary)
    err = hx.reset()
    if err:  # you can check if the reset was successful
        print(
            'ERROR-1: Reset failed. Not ready, please check the Pins and wiring up\n')
    else:
        print('GOOD-1: Calibrating, please wait')

    # Read data several, or only one, time and return mean value
    # argument "readings" is not required default value is 30
    data = hx.get_raw_data_mean(readings=30)

    if data:  # always check if you get correct value or only False
        print('GOOD-2: Raw data: {}'.format(data))
    else:
        print('ERROR-2: invalid data \n')

    # measure tare and save the value as offset for current channel
    # and gain selected. That means channel A and gain 64
    hx.zero(readings=30)

    # Read data several, or only one, time and return mean value.
    # It subtracts offset value for particular channel from the mean value.
    # This value is still just a number from HX711 without any conversion
    # to units such as grams or kg.
    data = hx.get_data_mean(readings=30)

    if data:  # always check if you get correct value or only False
        # now the value is close to 0
        print('GOOD-3: Data subtracted by offset but still not converted to any unit:',
              data)
    else:
        print('ERROR-3: invalid data')

    if res['tags']['sensor_ratio'] is None:
        # In order to calculate the conversion ratio to some units, in my case I want grams,
        # you must have known weight.
        input('\nPut known weight on the scale and then press Enter and WAIT')
        data = hx.get_data_mean(readings=30)
        if data:
            print('Mean value from HX711 subtracted by offset:', data)
            known_weight_grams = input(
                'Write how many grams it was and press Enter: ')
            try:
                value = float(known_weight_grams)
                print('Input: {} grams'.format(value))
            except ValueError:
                print('ERROR-4: Expected integer or float and I have got:\n',
                      known_weight_grams)

            # set scale ratio for particular channel and gain which is
            # used to calculate the conversion to units. Required argument is only
            # scale ratio. Without arguments 'channel' and 'gain_A' it sets
            # the ratio for current channel and gain.
            ratio = data / value  # norminised input
            # weight = data /ratio
            print('GOOD-5: Ratio is set. ')
        else:
            raise ValueError(
                'ERROR-5: Cannot calculate mean value. Try debug mode. \n')
    else:
        ratio =float(res['tags']['sensor_ratio'])
        print('GOOD-4: Read ration from store')
    return hx, ratio


def read(hx, ratio, feq):

    reading = hx.get_data_mean(feq)
    # print('raw: {:.2f}'.format(reading))
    f_nominal = reading/ratio
    # print('Nominal: {:.2f} g'.format(f_nominal))
    return reading, f_nominal


def reads(hx, res):
    print('\nplease clear the rack')
    print("Now, I will read data in infinite loop. To exit press 'CTRL + C'")
    input('Press Enter to begin reading')
    print('Current weight on the scale in kilo grams is: ')
    ratio = float(res["tags"]["sensor_ratio"])
    fr_filter = float(res["tags"]["filter"])
    feq = int(res["tags"]["feq"])
    try:
        while True:
            # timer = 0
            # reading_start, f_nominal_start = read(hx, ratio, feq)
            # time_start = time.time()
            # time_ones = []
            # while timer < interval:
            #     time_start_one = time.time()
            #     reading, f_nominal = read(hx, ratio, feq)
            #     timer_one = time.time()-time_start_one
            #     time_ones.append(timer_one)
            #     timer = time.time()-time_start
            #     if timer_one > interval:
            #         print("ERROR: reading is not fast enough, please check 'feq'")
            # f_time_one = sum(time_ones)/len(time_ones)
            # f_nominal_delta = f_nominal-f_nominal_start
            # f_flowrate = f_nominal_delta/timer

            reading_start, f_nominal_start = read(hx, ratio, feq)
            time_start = time.time()
            reading, f_nominal = read(hx, ratio, feq)
            f_time_one = time.time()-time_start
            f_nominal_delta = f_nominal-f_nominal_start
            f_flowrate = f_nominal_delta/f_time_one

            # while timer < interval:
            #     time_start_one = time.time()
            #     reading, f_nominal = read(hx, ratio, feq)
            #     timer_one = time.time()-time_start_one
            #     time_ones.append(timer_one)
            #     timer = time.time()-time_start
            #     if timer_one > interval:
            #         print("ERROR: reading is not fast enough, please check 'feq'")
            # f_time_one = sum(time_ones)/len(time_ones)
            # f_nominal_delta = f_nominal-f_nominal_start
            # f_flowrate = f_nominal_delta/timer

            # check, avoid noisy

            if abs(f_flowrate)<fr_filter:
                res["fields"]["f_raw"] = reading
                res["fields"]["f_nominal"] = f_nominal
                res["fields"]["f_flowrate"] = f_flowrate
                res["fields"]["f_simplerate"] = feq/f_time_one
            


                print('Weight: {:.2f} g, Flowrate: {:.2f} g/sec, Simplerate: {:.2f} Hz'.format(
                    f_nominal, f_flowrate, res["fields"]["f_simplerate"] ))

                data=[]
                data.append(res)

                if res['tags']['MQTT_CHECK']:
                    publish.single(res['tags']['MQTT_TOPIC'], json.dumps(
                        data), hostname=res['tags']['MQTT_SERVER'])
            res["fields"] = {}
    except (KeyboardInterrupt, SystemExit):
        print('Bye :)')


if __name__ == '__main__':
    hx, res = load_setup()
    reads(hx, res)
