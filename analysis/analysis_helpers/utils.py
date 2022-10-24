import datetime
import pytz

tz = pytz.timezone('UTC')

colours = {
    'blue' : '\033[94m',
    'green' : '\033[92m',
    'yellow' : '\033[93m',
    'red' : '\033[91m',
    'pink' : '\033[95m',
}
endColour = '\033[0m'

def current_datetime():
    return datetime.datetime.now(tz)

def printWithDate(s, colour = None, **kwargs):
    if colour is None:
        print(f"{datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')} {s}", **kwargs)
    else:
        print(f"{datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}{colours[colour]} {s}{endColour}", **kwargs)
