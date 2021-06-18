import cv2
import glob
import os

pre = "./preprocessing/*.jpg"
out = "./target"
name = "3"

for i, path in enumerate(glob.glob(pre)):
    print("before:", path)
    rename = out + "/{}_{}.jpg".format(name, str(i).zfill(3))
    print("after:", rename)
    os.rename(path, rename)
