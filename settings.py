import os


# Camera related
ROW = 1200
COLUMN = 1920
RGB_INDEX = 3

RED_RATIO = 0.2989
GREEN_RATIO = 0.5870
BLUE_RATIO = 0.1140

IMAGE_FORMAT = ".raw"

# Image related
CONTOUR_LEVEL = 15
PIXEL_RESOLUTION = 0.049

DROP_OFF_LOW = 0
DROP_OFF_HIGH = 1

FRAME_NUM = 15

# Image file Related
FILE_LIST = sorted(os.listdir("/Users/chen/Desktop/github/TEM_emittance/emittance_2"))
FILE_PATH = "/Users/chen/Desktop/github/TEM_emittance/emittance_2/"

# Solenoid voltage vs. magnetic field file related
TEM_SOL_VOLTAGE_LOC = "/Users/chen/Desktop/github/TEM_emittance/TEM_C0_peakvsvoltage.xlsx"
SOL_CURRENT_COLUMN = 'current (A)'
SOL_VOLTAGE_COLUMN = 'Voltage (V)'
SOL_B_FIELD_COLUMN = 'Magnetic field (G)'

# Solenoid posiion magnetic field file related
TEM_SOL_FIELD_MAP_LOC = "/Users/chen/Desktop/github/TEM_emittance/TEM_C0_detail.xls"