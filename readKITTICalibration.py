import sys
import numpy as np

K2 = 'K_03'
D1 = 'D_02'
D2 = 'D_03'

if len(sys.argv) < 2:
    print("path to textfile containing KITTI calibration data needs to be provided as argument")

filename = sys.argv[1]

f = open(filename, "r")
lines = f.read().splitlines()

for line in lines:
    split = line.split()
    match split[0]:
        case 'S_02:':
            w = int(float(split[1]))
            h = int(float(split[2]))
        case 'K_02:':
            K1 = np.array((split[1:])).astype(np.double)
        case 'K_03:':
            K2 = np.array((split[1:])).astype(np.double)
        case 'D_02:':
            dist1 = np.array(split[1:]).astype(np.double)
        case 'D_03:':
            dist2 = np.array(split[1:]).astype(np.double)
        case _:
            print("unmatched")

K1 = np.reshape(K1, (3, 3))
K2 = np.reshape(K2, (3, 3))

print(f"K1:\n{K1}\n")
print(f"K2:\n{K2}\n")
print(f"dist1:\n{dist1}\n")
print(f"dist2:\n{dist2}\n")
print(f"shape: {(w, h)}")



f.close()

np.savez("intrinsicParametersKITTI.npz", K1=K1, dist1=dist1, K2=K2, dist2=dist2, w=w, h=h)