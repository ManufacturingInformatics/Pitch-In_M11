"""Play a sine signal."""
import argparse
import sys

import numpy as np
import sounddevice as sd


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

## parse arguments to control the signal and where it goes
parser = argparse.ArgumentParser(add_help=False)
# add option to list available options
parser.add_argument(
    '-l', '--list-devices', action='store_true', # store true or false, default true
    help='show list of audio devices and exit')
# get arguments
args, remaining = parser.parse_known_args()
# if the list devices was stated
if args.list_devices:
    # get devices and print list
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
## signal properties
# frequency. default 500 hz
parser.add_argument(
    'frequency', nargs='?', metavar='FREQUENCY', type=float, default=500,
    help='frequency in Hz (default: %(default)s)')
# device id
# use int_or_str function to determine what was given. Attempt to convert string to integer if possible
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='output device (numeric ID or substring)')
# amplitude
parser.add_argument(
    '-a', '--amplitude', type=float, default=0.2,
    help='amplitude (default: %(default)s)')
args = parser.parse_args(remaining)

start_idx = 0

try:
    # get the sample rate of the stated device 
    samplerate = sd.query_devices(args.device, 'output')['default_samplerate']

    # function for generating data in response to requests
    def callback(outdata, frames, time, status):
        # if there is a status update print it along the error output
        if status:
            print(status, file=sys.stderr)
        # get the global start_idx variable
        # saves recreating it constantly
        global start_idx
        # create a time vector based on the number of frames to process, the set samplerate and the starting index
        t = (start_idx + np.arange(frames)) / samplerate
        # reshape it into a column
        t = t.reshape(-1, 1)
        # generate magnitude vector for the corresponding time vector, set frequency and target amplitude
        outdata[:] = args.amplitude * np.sin(2 * np.pi * args.frequency * t)
        # increment the starting index of the next callback by the number of frames to process
        start_idx += frames

    ## set the output audio stream according to the settings
    # open the output stream according to the set settings
    # the callback generates the data in response to requests
    with sd.OutputStream(device=args.device, channels=1, callback=callback,
                         samplerate=samplerate):
        # print dummy text to 
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        # press return to exit
        # the audio callback occurs in the background
        input()
# in the event of a keyboard interrupt
# exit with a blank message
except KeyboardInterrupt:
    parser.exit('')
# generic exception handler
# print type of exception along with message
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
