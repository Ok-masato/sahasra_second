import glob
import os

target = "./ResNet50/5クラス各40枚_barbara_hw001_+h×w"
# 画像の先頭の番号
category = ["0", "1", "2", "3", "4"]
result_path = "./ResNet50/5クラス各40枚_barbara_hw001_+h×w/result.txt"


def write_result(_result_path, _text):
    with open(_result_path, mode='a') as f:
        f.write(_text)


def evaluation(target, category, result_path):
    files = os.listdir(target)
    ex_folder = [f for f in files if os.path.isdir(os.path.join(target, f))]

    # 実験フォルダー
    for ex in ex_folder:
        result = ""
        # クラス
        for i, include_cls in enumerate(glob.glob(target + "/" + ex + "/*")):
            ex_block = include_cls.split("\\")[0].split("/")[3]
            if i == 0:
                result += "{}\n".format(ex_block)

            cls_block = include_cls.split("\\")[1]
            cls_num = len(category)
            category_count = [0] * (cls_num + 1)

            # IDカウント
            for file in glob.glob(include_cls + "/*.jpg"):
                name_without_ext = os.path.splitext(os.path.basename(file))[0]
                category_block = name_without_ext.split("_")[0]

                flag = False
                for c in category:
                    if c == category_block:
                        flag = True
                        num = int(c)
                        category_count[num] += 1
                if not flag:
                    category_count[-1] += 1

            # 全表示
            # for j, count in enumerate(category_count):
            #     if j == 0:
            #         result += "{}: id_{} {}".format(cls_block, j, count)
            #     else:
            #         if j == int(len(category_count)-1):
            #             result += ", etc {}\n".format(count)
            #         else:
            #             result += ", id_{} {}".format(j, count)

            for j, count in enumerate(category_count):
                if j == 0:
                    result += "{}: ".format(cls_block)
                    if count != 0:
                        result += "id_{} {}, ".format(j, count)
                else:
                    if j == int(len(category_count) - 1):
                        if count != 0:
                            result += ", etc {}\n".format(count)
                        else:
                            result += "\n"
                    else:
                        if count != 0:
                            result += "id_{} {}, ".format(j, count)
                        # result += ", id_{} {}".format(j, count)
        write_result(result_path, result)


# evaluation(target, category, result_path)


