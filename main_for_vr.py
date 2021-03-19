import cv2
import argparse
from libs import DiffCreate, RefleshFolder
from PIL import Image, ImageTk
import numpy as np


'''

# 元もOpenCVのバージョンは4.0．0.23とかやった

input_dir:撮影画像
diff_dir：差分画像
out_dir：差分矩形を足した画像
output_txt_path:差分によって発見された矩形の位置（画像内の座標）情報
obj_dir : 画面に映っている物の画像が保存されている
back_img_dir : 差分の裏側の画像
obj_db : 物のデータベース
img_num : 画像の枚数（0が最初の背景になる）

mode : 過去のデータを読み込むかどうか
out_test : 教師データの場所

操作説明
ｓ：撮影
ｒ：関係する画像ファイルをすべてリフレッシュ
ｑ：終了
'''

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", type=str, default="first",
    help="mode: first or load"
)


args = parser.parse_args()

input_dir = "img/"
diff_dir = "Diff_img/"
out_dir = "Diff_detect_img/"
obj_dir = "obj_img/"
back_img_dir = "back_img/"
obj_db = "obj_db/"
trace_data = "trace_data/"

cam_num = 3
# こうすけ用パス
start_path = "./VR/"



class App:
    def __init__(self, start_path,
                 input_dir, diff_dir, out_dir, obj_dir, back_img_dir, obj_db, trace_data, img_num, n):

        self.img_num = img_num
        input_dir = start_path + input_dir + "camera_{}/".format(n)
        diff_dir = start_path + diff_dir + "camera_{}/".format(n)
        out_dir = start_path + out_dir + "camera_{}/".format(n)
        obj_dir = start_path + obj_dir + "camera_{}/".format(n)
        back_img_dir = start_path + back_img_dir + "camera_{}/".format(n)
        obj_db = start_path + obj_db + "camera_{}/".format(n)
        trace_data = start_path + trace_data + "camera_{}/trace.txt".format(n)

        self.dc = DiffCreate.DiffCreate(input_dir, diff_dir, out_dir,
                                        obj_dir, back_img_dir, obj_db, trace_data, img_num)


cam_list = [cv2.VideoCapture(n) for n in range(cam_num)]
img_num_list = [int(0) for n in range(cam_num)]
app_list = [App(start_path, input_dir, diff_dir, out_dir, obj_dir,
                back_img_dir, obj_db, trace_data, 0, n) for n in range(cam_num)]
print(img_num_list)

while(True):
    cap_flag = False
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('s'):
        cap_flag = True

    for i, cam in enumerate(cam_list):
        ret, frame = cam.read()
        cv2.imshow('camera' + str(i), frame)
        # start_path = "./VR/"
        # input_dir = "img/"
        # ./VR/img/
        path = start_path + "{}camera_{}/{}.jpg".format(input_dir, i, img_num_list[i])

        if cap_flag:
            print(path)
            if img_num_list[i] == 0:
                cv2.imwrite(path, frame)
                img_num_list[i] = img_num_list[i] + 1
                print(img_num_list)
                app_list[i].dc.plus_num()
                print("Success capture(back).")
            else:
                cv2.imwrite(path, frame)
                done = app_list[i].dc.detect_contour()
                if done:
                    img_num_list[i] = img_num_list[i] + 1
                    app_list[i].dc.plus_num()
                print("Success this capture num ({}).".format(app_list[i].img_num))
            print(img_num_list)
        else:
            pass

for cam in cam_list:
    cam.release()
cv2.destroyAllWindows()
