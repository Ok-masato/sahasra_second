import cv2
import numpy as np
import os
import uuid


class DiffCreate:
    def __init__(self, input_dir, diff_dir, out_dir, obj_dir, back_img_dir, obj_db,  trace_data, num):
        self.input_dir = input_dir
        self.diff_dir = diff_dir
        self.out_dir = out_dir
        self.obj_dir = obj_dir
        self.back_img_dir = back_img_dir
        self.obj_db = obj_db
        self.num = num
        self.trace_data = trace_data
        self.objects = []

    def plus_num(self):
        self.num += 1

    def padding_position(self, x, y, w, h, p):
        return x - p, y - p, w + p * 2, h + p * 2

    # checkはbboxと同じ大きさで、初期値True
    def disappear_check(self, check, bbox):
        obj_del_list = []
        if self.objects:
            print("画面内にあったobjectの位置↓")
            print(self.objects)
        else:
            print("NO objects")
        print("動いた物の位置は→", bbox)
        print("消すかどうかは→", check)

        for j in range(len(self.objects)):
            for k in range(len(self.objects[j])):
                # print(self.objects[j][k])
                for i in range(len(bbox)):
                    point = 0
                    if bbox[i][0] == self.objects[j][k][0]:
                        point += 1
                    if bbox[i][1] == self.objects[j][k][1]:
                        point += 1
                    if bbox[i][2] == self.objects[j][k][2]:
                        point += 1
                    if bbox[i][3] == self.objects[j][k][3]:
                        point += 1

                    if point >= 3:
                        check[i] = False
                        print("消えたobjectが見つかりました。削除します。", self.objects[j][k], check)
                        obj_del_list.append([j, k])
                        # os.remove(self.objects[j][k][4])
                    else:
                        # check[i] = True
                        print("見つかりませんでした.", bbox[i], self.objects[j][k], check)

        print("削除前", self.objects)
        for l in range(len(obj_del_list)):
            try:
                del self.objects[obj_del_list[l][0]][obj_del_list[l][1]]
            except IndexError:
                pass
        print("削除後", self.objects)

        print("消すobjectはFalse、結果→", check)
        return check

    # 写真内に存在している物の場所と保存する画像の名前を返す
    def bbox2save_path(self, check, bbox):
        path_list = []
        print("--------------------------")
        print(check)
        print(bbox)
        for index in range(len(check)):
            if check[index]:
                obj_save_path = self.obj_dir + str(self.num) + "-" + str(self.num - 1) + "-" + str(index) + ".jpg"
                path_list.append([bbox[index][0], bbox[index][1], bbox[index][2], bbox[index][3], obj_save_path])
        return path_list

    def obj_pos_save(self, pos):
        self.objects.append(pos)

    def trim_img_save(self, name, img, pos, back_or_obj):
        tmp = img[pos[1]:pos[3], pos[0]:pos[2]]
        if back_or_obj == "o":
            cv2.imwrite(self.obj_dir + name, tmp)
        elif back_or_obj == "b":
            cv2.imwrite(self.back_img_dir + name, tmp)
        else:
            print("予期しない選択肢 : trim_img_save()")
            pass

    def obj_db_save(self, name, img, pos):
        tmp = img[pos[1]:pos[3], pos[0]:pos[2]]
        cv2.imwrite(self.obj_db + name, tmp)

    def trace_save(self, name, pos):
        with open(self.trace_data, mode='a') as f:
            tmp_line = self.input_dir + str(self.num) + ".jpg" + "," + self.obj_db + name + "," \
                       + str(pos[0]) + "," + str(pos[1]) + "," + str(pos[2]) + "," + str(pos[3])
            line = tmp_line.replace("/", "\\")
            f.write(line + "\n")
        print("trace_log:", line)

    def get_img_num(self):
        return self.num

    # アノテーション作成メソッド
    # def annotation_txt(bbox, basename):
    #     with open(anontation_txt_path, mode='a') as f:
    #         f.write(output_img_folder + str(basename) + " " + str(bbox[0]) + "," + str(bbox[1]) + "," + str(
    #             bbox[2]) + "," + str(bbox[3]) + "," + "0" + "\n")

    # 輪郭検出
    def detect_contour(self):

        back_img_path = self.input_dir + str(self.num - 1) + ".jpg"
        back_img = cv2.imread(back_img_path)

        obj_img_path = self.input_dir + str(self.num) + ".jpg"
        print("img_num : ", self.num)
        print("back_img : ", back_img_path)
        obj_img = cv2.imread(obj_img_path, 1)
        back_gray = cv2.cvtColor(back_img, cv2.COLOR_RGB2GRAY)
        obj_gray = cv2.cvtColor(obj_img, cv2.COLOR_RGB2GRAY)

        diff_img = back_gray.astype(int) - obj_gray.astype(int)
        diff_img_abs = np.abs(diff_img)

        abs_path = "./tmp_abs.jpg"
        cv2.imwrite(abs_path, diff_img_abs)

        diff_img = cv2.imread(abs_path)

        # ぼかし
        diff_img = cv2.cvtColor(cv2.GaussianBlur(diff_img, (5, 5), 0), cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("./bokashi.jpg", diff_img)
        # キャニー法を使った閾値処理
        diff_img = cv2.Canny(diff_img, threshold1=20, threshold2=110)
        # cv2.imwrite("./canny.jpg", diff_img)

        # 二値化されたグレースケール画像（RGBで表現されていない）
        # ret2, to2img = cv2.threshold(diff_img, 0, 255, cv2.THRESH_OTSU)
        ret2, to2img = cv2.threshold(diff_img, 50, 255, cv2.THRESH_BINARY)

        # 検出画像
        bg_diff_path = self.diff_dir + str(self.num) + "-" + str(self.num-1) + '.jpg'
        cv2.imwrite(bg_diff_path, to2img)

        # デバッグ用
        img = cv2.imread(bg_diff_path)
        contoured = img

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        se = np.ones((7, 7), dtype='uint8')
        img_close = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
        img_close = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
        img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

        morph_close_path = "./morph_close/" + str(self.num) + "-" + str(self.num-1) + '.jpg'
        cv2.imwrite(morph_close_path, img_close)

        # detect contour
        _, contours, hierarchy = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        area_list = []
        bbox_list = []

        # 差分に変化あり
        if contours:
            print("変化あり")
            for c in contours:
                area_list.append(cv2.contourArea(c))

            print("len", len(area_list))
            # 差分から物の矩形を抽出
            if area_list:
                for i in range(len(area_list)):
                    x, y, w, h = cv2.boundingRect(contours[i])
                    x, y, w, h = self.padding_position(x, y, w, h, 5)
                    # draw contour
                    x1 = x
                    y1 = y
                    w1 = w
                    h1 = h
                    c1 = contours[i]

                    cv2.drawContours(contoured, c1, -1, (0, 0, 255), 3)  # contour
                    bbox = [x1, y1, (x1 + w1), (y1 + h1)]
                    height, width = obj_img.shape[:2]
                    # 外形を超えた場合
                    for idx, coordinate in enumerate(bbox):
                        if (idx == 0 or idx == 1) and (coordinate < 0):
                            bbox[idx] = 0
                        if (idx == 2) and (coordinate > width):
                            bbox[idx] = width
                        if (idx == 3) and (coordinate > height):
                            bbox[idx] = height
                    print(bbox)
                    bbox_list.append(bbox)

                bbox_check_list = [True] * int(len(bbox_list))
                print(bbox_check_list)
                # 抽出した複数の物の場所を描画するから判定する（物同士が7割以上重なっていたら描画しない）
                for j in range(len(bbox_list)):
                    print("j:", j, bbox_list[j])
                    for k in range(len(bbox_list)):
                        if j != k:
                            h = bbox_list[j][3] - bbox_list[j][1]
                            w = bbox_list[j][2] - bbox_list[j][0]
                            rec = h * w
                            if (bbox_list[k][0] <= bbox_list[j][0]) and (bbox_list[j][0] <= bbox_list[k][2])\
                                    and ((bbox_list[k][1] <= bbox_list[j][1]) and (bbox_list[j][1] <= bbox_list[k][3])):
                                h1 = bbox_list[j][3] - bbox_list[j][1]
                                w1 = bbox_list[k][2] - bbox_list[j][0]
                                rec1 = h1 * w1
                                if rec1 / rec >= 0.7:
                                    bbox_check_list[j] = False
                                else:
                                    bbox_check_list[j] = True

                            if ((bbox_list[k][0] <= bbox_list[j][2]) and (bbox_list[j][2] <= bbox_list[k][2]))\
                                    and ((bbox_list[k][1] <= bbox_list[j][1]) and (bbox_list[j][1] <= bbox_list[k][3])):
                                h2 = bbox_list[j][3] - bbox_list[j][1]
                                w2 = bbox_list[j][2] - bbox_list[k][0]
                                rec2 = h2 * w2
                                if rec2 / rec >= 0.7:
                                    bbox_check_list[j] = False
                                else:
                                    bbox_check_list[j] = True

                            if ((bbox_list[k][1] <= bbox_list[j][1]) and (bbox_list[j][1] <= bbox_list[k][3])) \
                                    and ((bbox_list[k][0] <= bbox_list[j][1]) and (bbox_list[j][1] <= bbox_list[k][2])):
                                h3 = bbox_list[k][3] - bbox_list[j][1]
                                w3 = bbox_list[j][2] - bbox_list[j][0]
                                rec3 = h3 * w3
                                if rec3 / rec >= 0.7:
                                    bbox_check_list[j] = False
                                else:
                                    bbox_check_list[j] = True

                            if ((bbox_list[k][1] <= bbox_list[j][3]) and (bbox_list[j][3] <= bbox_list[k][3]))\
                                    and ((bbox_list[k][0] <= bbox_list[j][1]) and (bbox_list[j][1] <= bbox_list[k][2])):
                                h4 = bbox_list[j][3] - bbox_list[k][1]
                                w4 = bbox_list[j][2] - bbox_list[j][0]
                                rec4 = h4 * w4
                                if rec4 / rec >= 0.7:
                                    bbox_check_list[j] = False
                                else:
                                    bbox_check_list[j] = True

                print(bbox_check_list)
                if self.num > 1:
                    print("DISAPPEAR CHECK")
                    bbox_check_list = self.disappear_check(bbox_check_list, bbox_list)
                    print("検索削除後", self.objects)

                # 画像に物の矩形をトリミングして分類する
                for index in range(len(bbox_check_list)):
                    if bbox_check_list[index]:
                        # 物の画像をトリミングして保存する
                        tmp_path = "obj-{}-{}-{}.jpg".format(str(self.num), str(self.num - 1), str(index))
                        self.trim_img_save(tmp_path, obj_img, bbox_list[index], "o")
                        # 差分の裏側も保存する
                        tmp_path = "back-{}-{}-{}.jpg".format(str(self.num), str(self.num - 1), str(index))
                        self.trim_img_save(tmp_path, back_img, bbox_list[index], "b")

                        # 物の画像をトリミングして類似画像の検索、DBへ保存
                        u4 = str(uuid.uuid4())
                        # print(u4)
                        db_tmp = "{}-{}_{}.jpg".format(str(self.num), str(self.num - 1), u4)

                        self.obj_db_save(db_tmp, obj_img, bbox_list[index])
                        self.trace_save(db_tmp, bbox_list[index])
                        # similar_img.start(self.num, bbox_list[index])

                # 矩形の描画
                for index in range(len(bbox_check_list)):
                    if bbox_check_list[index]:
                        cv2.rectangle(contoured, (bbox_list[index][0], bbox_list[index][1]),
                                      (bbox_list[index][2], bbox_list[index][3]), (0, 255, 0), 3)  # rectangle contour
                        cv2.rectangle(obj_img, (bbox_list[index][0], bbox_list[index][1]),
                                      (bbox_list[index][2], bbox_list[index][3]), (0, 255, 0), 3)
                        # center_x, center_y = self.center_position(bbox_list[index])
                        # cv2.circle(obj_img, (center_x, center_y), 10, 255, -1)
                cv2.imwrite(self.out_dir + str(self.num) + "-" + str(self.num-1) + '.jpg', obj_img)
                # x,y centerデータの作成
                # self.out_txt(center_x, center_y, obj_img_path)

            pos_and_path_list = self.bbox2save_path(bbox_check_list, bbox_list)
            self.obj_pos_save(pos_and_path_list)

            print("Objects:", self.objects)

            return True

        # 差分で変化が見れなかったので、撮影した画像、差分画像、それにかかわる画像をを削除する
        else:
            print("変化なし")
            del_file_list = [obj_img_path, morph_close_path, bg_diff_path, abs_path]
            for del_file in del_file_list:
                print("差分に変化がなかったため削除します。:", del_file)
                os.remove(del_file)

            return False
