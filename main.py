import cv2
import argparse
from libs import DiffCreate, RefleshFolder
from tkinter import *
from PIL import Image, ImageTk
import human_detector
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
'''

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", type=str, default="first",
    help="mode: first or load"
)

args = parser.parse_args()

input_dir = "./img/"
diff_dir = "./Diff_img/"
out_dir = "./Diff_detect_img/"
obj_dir = "./obj_img/"
back_img_dir = "./back_img/"
obj_db = "./obj_db/target/"
trace_data = "./train_data/trace.txt"

img_num = 0

back_flag = False
human_flag = False
camera_scale = 1.

class App:
    def __init__(self, window, window_title,
                 input_dir, diff_dir, out_dir, obj_dir, back_img_dir, obj_db, trace_data, img_num, back_flag):

        self.window = window
        self.window.title(window_title)

        self.vcap = cv2.VideoCapture(0)
        self.width = self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.yolo = human_detector.YOLO()
        self.human_exist = [False, False, False, False]

        # カメラモジュールの映像を表示するキャンバスを用意する
        self.canvas = Canvas(self.window, width=self.width, height=self.height)
        self.canvas.grid(columnspan=3, column=0, row=0, sticky=W+E)

        self.img_num = img_num
        self.input_dir = input_dir
        self. back_flag = back_flag
        self.trace_data = trace_data

        self.dc = DiffCreate.DiffCreate(input_dir, diff_dir, out_dir,
                                        obj_dir, back_img_dir, obj_db, trace_data, img_num)

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
        # 保存用のイメージ
        tmp_image = image
        h, w = image.shape[:2]
        rh = int(h * camera_scale)
        rw = int(w * camera_scale)
        image = cv2.resize(image, (rw, rh))
        image = image[:, :, (2, 1, 0)]
        image = Image.fromarray(image)
        r_image, human_flag = self.yolo.detect_image(image)
        del self.human_exist[0]
        self.human_exist.append(human_flag)
        print(self.human_exist)

        if self.back_flag and self.check_capture():
            path = self.input_dir + str(self.img_num) + ".jpg"
            cv2.imwrite(path, tmp_image)
            done = self.dc.detect_contour()
            if done:
                self.img_num = self.img_num + 1
                self.dc.plus_num()
            print("Success this capture num ({}).".format(self.img_num))

        out_img = np.array(r_image)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(out_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
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
        _, frame = self.vcap.read()

        if args.mode == "first":
            if not self.back_flag:
                self.back_flag = True
                cv2.imwrite(path, frame)
                self.img_num = self.img_num + 1
                self.dc.plus_num()
                self.ref_btn['state'] = DISABLED
                print("Success capture(back).")

            # add
            else:
                cv2.imwrite(path, frame)
                done = self.dc.detect_contour()
                if done:
                    self.img_num = self.img_num + 1
                    self.dc.plus_num()
                print("Success this capture num ({}).".format(self.img_num))

        elif args.mode == "load":
            pass


App(Tk(), "Sahasra Difference Detector ", input_dir, diff_dir, out_dir, obj_dir, back_img_dir, obj_db, trace_data, img_num, back_flag)