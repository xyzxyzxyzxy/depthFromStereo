import numpy as np
import cv2 as cv
import sys

if len(sys.argv) < 2:
    print("Stereo pair index required")
    sys.exit()

INDEX = int(sys.argv[1])

fname = "stereoData" + str(INDEX) + ".npz"
data = np.load(fname)

F1 = data['F']
E1 = data['E']
R1_1 = data['R1']
t1 = data['t']
Q = data['Q']
roiL = data['roiL']
roiR = data['roiR']


print(f"F1:\n{F1}\nE1:\n{E1}\nR1_1:\n{R1_1}\nt1:\n{t1}\nQ:\n{Q}\nroiL:\n{roiL}\nroiR:\n{roiR}")