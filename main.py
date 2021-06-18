# conda install -c conda-forge opencv
import cv2
import argparse
from libs import DiffCreate, RefleshFolder
from tkinter import *
from PIL import Image, ImageTk
import human_detector
import numpy as np
import os

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
'''

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", type=str, default="first",
    help="mode: first or load"
)

args = parser.parse_args()

input_dir = "./img/"
if not os.path.exists(input_dir):
    os.mkdir(input_dir)

diff_dir = "./Diff_img/"
if not os.path.exists(diff_dir):
    os.mkdir(diff_dir)

out_dir = "./Diff_detect_img/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

obj_dir = "./obj_img/"
if not os.path.exists(obj_dir):
    os.mkdir(obj_dir)

back_img_dir = "./back_img/"
if not os.path.exists(back_img_dir):
    os.mkdir(back_img_dir)

obj_db = "./obj_db/"
if not os.path.exists(obj_db):
    os.mkdir(obj_db)

obj_db = "./obj_db/target/"
if not os.path.exists(obj_db):
    os.mkdir(obj_db)

trace_data = "./train_data/trace.txt"

img_num = 0
img_num2 = 0

back_flag = False
human_flag = False
human_flag2 = False
camera_scale = 1.

# 複数カメラ用
camera_num = 1
flag = True
captures = []


class App:
    def __init__(self, window, window_title, input_dir, diff_dir, out_dir, obj_dir,
                 back_img_dir, obj_db, trace_data, img_num, img_num2, back_flag, captures):

        self.window = window
        self.window.title(window_title)

        self.vcap = cv2.VideoCapture(0)
        self.vcap2 = cv2.VideoCapture(1)
        self.width = self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width2 = self.vcap2.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height2 = self.vcap2.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.yolo = human_detector.YOLO()
        self.human_exist = [False, False, False, False]

        # カメラモジュールの映像を表示するキャンバスを用意する
        self.canvas = Canvas(self.window, width=640, height=480)
        self.canvas2 = Canvas(self.window, width=640, height=480)
        self.canvas.grid(columnspan=3, column=0, row=0, sticky=W + E)
        self.canvas2.grid(columnspan=3, column=3, row=0, sticky=W + E)

        self.img_num = img_num
        self.img_num2 = img_num2
        self.input_dir = input_dir
        self.back_flag = back_flag
        self.trace_data = trace_data

        self.dc = DiffCreate.DiffCreate(input_dir, diff_dir, out_dir,
                                        obj_dir, back_img_dir, obj_db, trace_data, img_num, 1)
        self.dc2 = DiffCreate.DiffCreate(input_dir, diff_dir, out_dir, obj_dir, back_img_dir, obj_db, trace_data,
                                         img_num, 2)

        # 撮影ボタン
        self.cap_btn = Button(self.window, text="Capture")
        self.cap_btn.grid(column=0, row=1, padx=10, pady=10)
        self.cap_btn.configure(command=self.capture)

        self.ref_btn = Button(self.window, text="Reflesh")
        self.ref_btn.grid(column=1, row=1, padx=10, pady=10)
        self.ref_btn.configure(command=self.reflesh)

        # 終了ボタン
        self.close_btn = Button(self.window, text="Close")
        self.close_btn.grid(column=2, row=1, padx=10, pady=10)
        self.close_btn.configure(command=self.destructor)

        # update()関数を15ミリ秒ごとに呼び出し、
        # キャンバスの映像を更新する
        self.delay = 15
        self.update()

        self.window.mainloop()

    def check_capture(self):
        if (self.human_exist[0] == True) and (self.human_exist[1] == False) and (self.human_exist[2] == False) \
                and (self.human_exist[3] == False):
            return True
        else:
            return False

    # キャンバスに表示されているカメラモジュールの映像を
    # 15ミリ秒ごとに更新する
    def update(self):
        ret, image = self.vcap.read()
        ret2, image2 = self.vcap2.read()
        # 保存用のイメージ
        tmp_image = image
        tmp_image2 = image2
        h, w = image.shape[:2]
        rh = int(h * camera_scale)
        rw = int(w * camera_scale)

        image = cv2.resize(image, (rw, rh))
        image = image[:, :, (2, 1, 0)]
        image = Image.fromarray(image)

        image2 = cv2.resize(image2, (rw, rh))
        image2 = image2[:, :, (2, 1, 0)]
        image2 = Image.fromarray(image2)

        r_image, human_flag = self.yolo.detect_image(image)
        r_image2, human_flag2 = self.yolo.detect_image(image2)

        del self.human_exist[0]
        self.human_exist.append(human_flag)
        print(self.human_exist)

        if self.back_flag and self.check_capture():
            path = self.input_dir + str(self.img_num) + ".jpg"
            path2 = self.input_dir + str(self.img_num2) + "-2.jpg"

            cv2.imwrite(path, tmp_image)  # 画像ファイル(input_dir)に保存
            cv2.imwrite(path2, tmp_image2)

            done = self.dc.detect_contour()
            done2 = self.dc2.detect_contour()
            if done:
                self.img_num = self.img_num + 1
                self.dc.plus_num()

            if done2:
                self.img_num2 = self.img_num2 + 1
                self.dc2.plus_num()

            print("Success this capture num ({}).".format(self.img_num))

        out_img = np.array(r_image)
        out_img2 = np.array(r_image2)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(out_img))
        self.photo2 = ImageTk.PhotoImage(image=Image.fromarray(out_img2))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.canvas2.create_image(3, 3, image=self.photo2, anchor=NW)

        self.window.after(self.delay, self.update)

    # リフレッシュボタンの処理
    def reflesh(self):
        self.ref_btn['state'] = DISABLED
        RefleshFolder.reflesh()

    # Closeボタンの処理
    def destructor(self):
        print("Quit.")
        self.window.destroy()
        self.vcap.release()
        self.yolo.close_session()

    # 撮影ボタンの処理
    def capture(self):
        path = self.input_dir + str(self.img_num) + ".jpg"
        path2 = self.input_dir + str(self.img_num2) + "-2.jpg"

        _, frame = self.vcap.read()
        _, frame2 = self.vcap2.read()

        if args.mode == "first":
            if not self.back_flag:
                self.back_flag = True
                cv2.imwrite(path, frame)  # 画像ファイル(input_dir)に保存
                cv2.imwrite(path2, frame2)
                self.img_num = self.img_num + 1
                self.img_num2 = self.img_num2 + 1
                self.dc.plus_num()
                self.dc2.plus_num()
                self.ref_btn['state'] = DISABLED
                print("Success capture(back).")

            # add
            else:
                cv2.imwrite(path, frame)  # 画像ファイル(input_dir)に保存
                cv2.imwrite(path2, frame2)
                done = self.dc.detect_contour()
                done2 = self.dc2.detect_contour()
                if done:
                    self.img_num = self.img_num + 1
                    self.dc.plus_num()

                if done2:
                    self.img_num2 = self.img_num2 + 1
                    self.dc2.plus_num()
                print("Success this capture num ({}).".format(self.img_num))

        elif args.mode == "load":
            pass


App(Tk(), "Sahasra Difference Detector ", input_dir, diff_dir, out_dir, obj_dir, back_img_dir, obj_db, trace_data,
    img_num, img_num2, back_flag, captures)
