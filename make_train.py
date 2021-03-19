import os
import glob
from libs import AutomaticClassification as ac

annotation_path = "./train_data/annotation.txt"
label_path = "./train_data/label.txt"
trace_path = "./train_data/trace.txt"
# 分類対象のフォルダ
target = "./obj_db/target"
# 分類クラスのフォルダ名
result_folder = "/class_"
# 分類クラスのフォルダの頭
output_path = "./obj_db"
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


if __name__ == "__main__":

    ac = ac.AutomaticClassification(target, output_path, result_folder, model_name)
    ac.import_img()
    ac.start_classification()

    train_list = []
    label_list = []

    files = os.listdir(output_path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(output_path, f))]
    label_list = [i for i in files_dir if "class_" in i]
    print(label_list)

    with open(trace_path) as f:
        for s_line in f:
            block = s_line.split(",")
            print(block)
            original_img_path = block[0]
            xmin = block[2]
            ymin = block[3]
            xmax = block[4]
            ymax = block[5].replace("\n", "")

            trim_name = block[1].split("\\")[3]
            print(trim_name)

            tmp_train = original_img_path + "," + trim_name + "," \
                        + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax)
            train_list.append(tmp_train)

        print(train_list)

    for cls_id_num, cls_id in enumerate(label_list):
        for target_path in glob.glob(output_path + "/" + cls_id + "/*.jpg"):
            basename = os.path.basename(target_path)
            cp_train = train_list
            for i, _train in enumerate(cp_train):
                if basename in _train:
                    tmp = "," + basename
                    print(tmp)
                    train_list[i] = train_list[i].replace(tmp, " ")
                    train_list[i] = train_list[i] + "," + str(cls_id_num)

    print(train_list)

    with open(label_path, mode='w') as f:
        for label in label_list:
            print(label)
            f.write(label+"\n")

    with open(annotation_path, mode='w') as f:
        for train in train_list:
            print(train)
            f.write(train+"\n")
