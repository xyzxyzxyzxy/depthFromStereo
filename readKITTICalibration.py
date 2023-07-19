import sys
import numpy as np

if len(sys.argv) < 2:
    print("path to textfile containing KITTI calibration data needs to be provided as argument")

filename = sys.argv[1]

f = open(filename, "r")
lines = f.read().splitlines()

for line in lines:
    split = line.split()
    match split[0]:
        case 'S_00:':
            w = int(float(split[1]))
            h = int(float(split[2]))
        case 'K_00:':
            K1 = np.array((split[1:])).astype(np.double)
        case 'D_00:':
            dist1 = np.array(split[1:]).astype(np.double)
        case 'R_00:':
            R1 = np.array((split[1:])).astype(np.double)
        case 'T_00:':
            T1 = np.array((split[1:])).astype(np.double)
        case 'R_rect_00:':
            Rrect1 = np.array((split[1:])).astype(np.double)
        case 'P_rect_00:':
            Prect1 = np.array((split[1:])).astype(np.double)
        case 'K_01:':
            K2 = np.array((split[1:])).astype(np.double)
        case 'D_01:':
            dist2 = np.array(split[1:]).astype(np.double)
        case 'R_01:':
            R2 = np.array((split[1:])).astype(np.double)
        case 'T_01:':
            T2 = np.array((split[1:])).astype(np.double)
        case 'R_rect_01:':
            Rrect2 = np.array((split[1:])).astype(np.double)
        case 'P_rect_01:':
            Prect2 = np.array((split[1:])).astype(np.double)
        case _:
            pass

K1 = np.reshape(K1, (3, 3))
K2 = np.reshape(K2, (3, 3))
R1 = np.reshape(R1, (3, 3))
R2 = np.reshape(R2, (3, 3))
Rrect1 = np.reshape(Rrect1, (3, 3))
Prect1 = np.reshape(Prect1, (3, 4))
Rrect2 = np.reshape(Rrect1, (3, 3))
Prect2 = np.reshape(Prect1, (3, 4))

print(f"K1:\n{K1}\n")
print(f"K2:\n{K2}\n")
print(f"dist1:\n{dist1}\n")
print(f"dist2:\n{dist2}\n")
print(f"R1:\n{R1}\nT1:\n{T1}")
print(f"R2:\n{R2}\nT2:\n{T2}")
print(f"shape: {(w, h)}")

f.close()

np.savez("intrinsicParametersKITTI.npz", K1=K1, dist1=dist1, K2=K2, dist2=dist2, R1=R1, T1=T1, R2=R2, T2=T2, w=w, h=h)

#GET ALL THE DATA TO PERFORM COMPARISON WITH RECTIFY STEREO RESULTS
with open("kittiDataHR.txt", "w") as f:
    f.write("K1\n")
    np.savetxt(f, K1)
    f.write("dist1\n")
    np.savetxt(f, dist1)
    f.write("R1\n")
    np.savetxt(f, R1)
    f.write("T1\n")
    np.savetxt(f, T1)
    f.write("Rot_rect1\n")
    np.savetxt(f, Rrect1)
    f.write("Proj_rect1\n")
    np.savetxt(f, Prect1)
    f.write("K2\n")
    np.savetxt(f, K2)
    f.write("dist2\n")
    np.savetxt(f, dist2)
    f.write("R2\n")
    np.savetxt(f, R2)
    f.write("T2\n")
    np.savetxt(f, T2)
    f.write("Rot_rect2\n")
    np.savetxt(f, Rrect2)
    f.write("Proj_rect2\n")
    np.savetxt(f, Prect2)