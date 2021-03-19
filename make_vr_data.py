import os
import glob
from libs import AutomaticClassification as ac

# カメラ数
cam_num = 3
# こうすけ用パス
start_path = "./VR/"
out_file = "out.txt"
label_file = "label.txt"
trace_file = "trace.txt"
trace_folder = "trace_data/"
# 分類対象のフォルダ
target = "obj_db/"
# 分類クラスのフォルダ名
result_folder = "class_"
# 分類クラスのフォルダの頭
output_path = "obj_db"
# 画像のパス
img_paths = []
# 分類に使用するモデル
model_name = "resnet50"


'''
    trace_data:
    original_img_path, trimming_img_path(label名を含まないフルパスのファイル名), bbox
                                ・
                                ・
                                ・ 
'''
def out_txt(out_txt_path, center_x, center_y, img_path):
    with open(out_txt_path, mode='a') as f:
        f.write(str(img_path) + " " + str(center_x) + "," + str(center_y) + "\n")


def center_position(xmin, ymin, xmax, ymax):
    center_x = (int(xmin) + int(xmax)) / 2
    center_y = (int(ymin) + int(ymax)) / 2
    return int(center_x), int(center_y)


if __name__ == "__main__":

    # エラー回避のためのオブジェクト生成
    ac = ac.AutomaticClassification("", "", "", model_name)

    for num in range(cam_num):

        camera_folder = "camera_{}".format(num)
        _target = start_path + target + camera_folder
        _output_path = start_path + output_path + "/" + camera_folder
        ac.update_paramater(_target, _output_path, "/" + result_folder)
        ac.import_img()
        ac.start_classification()

        train_list = []
        label_list = []

        files = os.listdir(_output_path)
        files_dir = [f for f in files if os.path.isdir(os.path.join(_output_path, f))]
        label_list = [i for i in files_dir if "class_" in i]
        trace_path = start_path + trace_folder + camera_folder + "/" + trace_file

        with open(trace_path) as f:
            for s_line in f:
                block = s_line.split(",")
                original_img_path = block[0]
                xmin = block[2]
                ymin = block[3]
                xmax = block[4]
                ymax = block[5].replace("\n", "")

                center_x, center_y = center_position(xmin, ymin, xmax, ymax)

                trim_name = block[1].split("\\")[4]
                tmp_train = original_img_path + "," + trim_name + "," \
                            + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," \
                            + str(center_x) + "," + str(center_y)
                train_list.append(tmp_train)

        for cls_id_num, cls_id in enumerate(label_list):
            for target_path in glob.glob(_output_path + "/" + cls_id + "/*.jpg"):
                basename = os.path.basename(target_path)
                cp_train = train_list
                for i, _train in enumerate(cp_train):
                    if basename in _train:
                        tmp = "," + basename
                        train_list[i] = train_list[i].replace(tmp, "")
                        train_list[i] = train_list[i] + "," + str(cls_id_num)

        label_path = start_path + trace_folder + camera_folder + "/" + label_file

        with open(label_path, mode='w') as f:
            for label in label_list:
                f.write(label+"\n")

        out_path = start_path + trace_folder + camera_folder + "/" + out_file

        with open(out_path, mode='w') as f:
            for train in train_list:
                f.write(train+"\n")
