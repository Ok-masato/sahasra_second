import cv2
import os
import itertools
import glob
import shutil
import sys
import uuid
import time

# SIFT
# 使えるOpenCVのバージョン
# pip install opencv-python==3.3.0.10 opencv-contrib-python==3.3.0.10


# flag True:降順, False:昇順 数値をもとにソートされたインデックスのリストを返す
# 例：[254, 34, 99] → [1, 2, 3] (昇順)

trace_data = r"C:\Users\ryodai.hamasaki\PycharmProjects\Smart\train_data\trace.txt"


def trace_write(original_img_path, trim_img_path, bbox):
    with open(trace_data, mode='a') as f:
        tmp_line = str(original_img_path) + "," + trim_img_path + "," + str(bbox[0]) + "," + str(bbox[1])+ "," + str(bbox[2])+ "," + str(bbox[3])
        line = tmp_line.replace("/", "\\")
        f.write(line + "\n")

def score_sort(target_list, flag):
    indices = [*range(len(target_list))]
    sorted_indices = sorted(indices, key=lambda i: -target_list[i], reverse=flag)
    sorted_num = [target_list[i] for i in sorted_indices]
    return sorted_indices


def prepare_score(n, _score):
    for i in range(n):
        _score.append([i, 0, 0, 0, 0])
    return _score


def calsulate_score(mode, sorted_indices, _score):
    if mode == "Max":
        for i in range(len(sorted_indices)):
            # 配列番号、得点
            tmp_pos = sorted_indices[i]
            tmp_score = -1 * (i+1)
            for j in range(len(_score)):
                if j == tmp_pos:
                    _score[tmp_pos][1] = tmp_score

    elif mode == "Min":
        for i in range(len(sorted_indices)):
            tmp_pos = sorted_indices[i]
            tmp_score = 2 * (i+1)
            for j in range(len(_score)):
                if j == tmp_pos:
                    _score[tmp_pos][2] = tmp_score

    elif mode == "Avg":
        for i in range(len(sorted_indices)):
            tmp_pos = sorted_indices[i]
            tmp_score = 1 * (i+1)
            for j in range(len(_score)):
                if j == tmp_pos:
                    _score[tmp_pos][3] = tmp_score

    elif mode == "Len":
        for i in range(len(sorted_indices)):
            tmp_pos = sorted_indices[i]
            tmp_score = 3 * (i+1)
            for j in range(len(_score)):
                if j == tmp_pos:
                    _score[tmp_pos][4] = tmp_score

    return _score


def result(_score, _name_list):
    sml_img_name = ""
    total = 0
    print("-------------------------------------------得点一覧-------------------------------------------")
    for i in range(len(_score)):
        tmp_total = sum(_score[i][1:])/len(_score)
        subdir = os.path.basename(os.path.dirname(_name_list[_score[i][0]]))
        filename = os.path.basename(_name_list[_score[i][0]])
        print("File: ./obj_db/{}/{}  Total score: {}".format(subdir, filename, tmp_total))
        if tmp_total >= total:
            total = tmp_total
            sml_img_name = _name_list[_score[i][0]]
    return sml_img_name, total


def start(num, bbox):
    print("-------------------------------------------類似画像探索開始-------------------------------------------")
    print("元画像: ./img/"+str(num)+".jpg")
    print("位置：", bbox)
    IMG_DIR = r'C:\Users\ryodai.hamasaki\PycharmProjects\Smart\obj_db'
    for target_file in glob.glob(IMG_DIR + "/*.jpg"):

        IMG_SIZE = (100, 100)

        if target_file:
            target_img_path = target_file
        else:
            print("There is no comparison target.")
            break

        # グレースケールのほうが精度が悪かった
        # target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(target_img_path)

        target_img = cv2.resize(target_img, IMG_SIZE)

        bf = cv2.BFMatcher()
        sift = cv2.xfeatures2d.SIFT_create()

        (target_kp, target_des) = sift.detectAndCompute(target_img, None)
        print('TARGET_FILE: {}'.format(os.path.basename(target_file)))
        dist = []

        files = glob.glob(IMG_DIR + "/**/*.jpg", recursive=True)

        for file in files:
            # time.sleep(1)
            f_basename = os.path.basename(file)
            t_basename = os.path.basename(target_img_path)
            if f_basename != t_basename:
                comparing_img_path = file
                try:
                    # グレースケールのほうが精度が悪かった
                    # comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
                    comparing_img = cv2.imread(comparing_img_path)
                    comparing_img = cv2.resize(comparing_img, IMG_SIZE)
                    (comparing_kp, comparing_des) = sift.detectAndCompute(comparing_img, None)

                    # matches = bf.match(target_des, comparing_des)
                    # マッチング（K=2個）させる
                    # KNN（K-Nearest Neighbor algorithm）は、探索空間から最近傍のラベルをK個選択し、多数決でクラスラベルを割り当てるアルゴリズム
                    matches = bf.knnMatch(target_des, comparing_des, k=2)
                    tmp = []

                    for i, pair in enumerate(matches):
                        try:
                            m, n = pair
                            # 近いK個の特徴点が存在する際に、より大きい方の特徴点（n.distance）の距離を少し小さくしても、小さい方の特徴点の距離(m.distance)が小さければ採択する
                            # distanceが小さいほど似ている
                            # print("distance:",m.distance, n.distance)
                            if m.distance < 0.7 * n.distance:
                                tmp.append(m.distance)
                                tmp.sort()

                        except ValueError:
                            print(ValueError)
                            break

                    # print(tmp)
                    dist.append([file, tmp])

                except cv2.error:
                    pass

        # distanceの最大値、最小値、平均値、個数を計算する
        dist_cal = []

        for i in range(len(dist)):
            for j in range(len(dist[i])):
                dis_avg = 0
                dis_max = 0
                dis_min = 0
                dis_len = 0

                if j == 0:
                    print("file: {}".format(dist[i][0]))
                else:
                    if dist[i][1]:
                        flat_tmp = list(itertools.chain.from_iterable(dist[i][1:]))
                        dis_avg = sum(flat_tmp) / len(flat_tmp)
                        dis_max = max(flat_tmp)
                        dis_min = min(flat_tmp)
                        dis_len = len(flat_tmp)

                        if dis_len <= 1:
                            print("類似点が少ないため評価に含まません。 (feature == 1)  -- Max: {}, Min: {}, Avg: {}, len: {} \n"
                                  .format(int(dis_max), int(dis_min), int(dis_avg), dis_len))
                        else:
                            print("類似度から計算した得点：  -- Max: {}, Min: {}, Avg: {}, len: {} \n"
                                  .format(int(dis_max), int(dis_min), int(dis_avg), dis_len))
                            dist_cal.append([dist[i][0], int(dis_max), int(dis_min), int(dis_avg), dis_len])
                    else:
                        print("類似点はありませんでした。 \n")

        '''
        計算結果からカテゴリー分けを行う
            得点について(計算結果の個数nをかける)
                最大値が最も大きい　－1 * n
                最小値が最も小さい　＋2 * n
                平均値が最も小さい　＋1 * n
                個数が最も多い　　　＋3 * n
        '''

        print("-------------------------------------------類似度の結果-------------------------------------------")
        for dis_cal_tmp in dist_cal:
            print("Name: {}, Max: {}, Min: {}, Avg: {}, Len: {}".format(dis_cal_tmp[0], dis_cal_tmp[1],
                                                                        dis_cal_tmp[2], dis_cal_tmp[3], dis_cal_tmp[4]))

        # 項目ごとに分解する
        name_list = []
        max_list = []
        min_list = []
        avg_list = []
        len_list = []
        for i in range(len(dist_cal)):
            name_list.append(dist_cal[i][0])
            max_list.append(dist_cal[i][1])
            min_list.append(dist_cal[i][2])
            avg_list.append(dist_cal[i][3])
            len_list.append(dist_cal[i][4])

        max_idx = score_sort(max_list, True)
        min_idx = score_sort(min_list, False)
        avg_idx = score_sort(avg_list, False)
        len_idx = score_sort(len_list, True)

        print("-------------------------------------------スコアの準備-------------------------------------------")
        score = []

        '''
        score = [
                    [name_index, max, min, avg, len], # name_index:0番目の得点、最大値の得点、最小値の得点、平均値の得点、長さの得点
                    [,,],
                    [],
                ]
        
        '''
        score = prepare_score(len(max_idx), score)
        print(score)

        print("-------------------------------------------類似度の計算-------------------------------------------")
        score = calsulate_score("Max", max_idx, score)
        print("MAX", score)
        score = calsulate_score("Min", min_idx, score)
        print("Min", score)
        score = calsulate_score("Avg", avg_idx, score)
        print("Avg", score)
        score = calsulate_score("Len", len_idx, score)
        print("Len", score)

        similar, total_score = result(score, name_list)

        original_img_path = r"C:\Users\ryodai.hamasaki\PycharmProjects\Smart\img\\" + str(num) + ".jpg"
        ta_file_name = os.path.basename(target_file)

        if similar and (not total_score == 5.0):
            s_dirname = os.path.dirname(similar)
            si_subdir = os.path.basename(os.path.dirname(similar))
            si_filename = os.path.basename(similar)
            print("最も似ている画像:  ./obj_db/{}/{}".format(si_subdir, si_filename))

            # 最も似ている画像のあるフォルダに移動する
            try:
                shutil.move(target_file, s_dirname)
            except Exception:
                pass

        else:
            u4 = str(uuid.uuid4())
            new_obj_dir = IMG_DIR + "/" + u4
            print(new_obj_dir)
            os.mkdir(new_obj_dir)
            shutil.move(target_file, new_obj_dir)
            # trace_write(original_img_path, new_obj_dir + "\\" + ta_file_name, bbox)
        print("-------------------------------------------------------------------------------------------------- \n")

# 1枚ずつしか教師データ用が作れない
# start(1, 1)
