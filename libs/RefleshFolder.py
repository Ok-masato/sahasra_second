import os
import glob

input_dir = "./img/"
diff_dir = "./Diff_img/"
out_dir = "./Diff_detect_img/"
obj_dir = "./obj_img/"
back_img_dir = "./back_img/"
obj_db = "./obj_db/"
dilate_dir = "./dilate/"


def reflesh():
    del_folder_list = [input_dir, diff_dir, out_dir, obj_dir, back_img_dir, obj_db, dilate_dir]
    print("-------------------------------------リフレッシュ開始-------------------------------------")

    for folders in del_folder_list:
        for file in glob.glob(folders + "/*.jpg") + glob.glob(folders + "/**/*.jpg"):
            print("ファイル:", file)
            os.remove(file)

    for folder in glob.glob(obj_db + "/**"):
        if "target" in folder:
            pass
        else:
            print("フォルダー:", folder)
            os.rmdir(folder)

    print("-------------------------------------リフレッシュ終了-------------------------------------")


def reflesh_vr(head_path):
    del_folder_list = [head_path + input_dir, head_path + diff_dir, head_path + out_dir, head_path + obj_dir,
                       head_path + back_img_dir, head_path + obj_db, head_path + dilate_dir]
    print("-------------------------------------リフレッシュ開始-------------------------------------")

    for folders in del_folder_list:
        for file in glob.glob(folders + "/*.jpg") + glob.glob(folders + "/**/*.jpg"):
            print("ファイル:", file)
            os.remove(file)

    for folder in glob.glob(obj_db + "/**"):
        print("フォルダー:", folder)
        os.rmdir(folder)

    print("-------------------------------------リフレッシュ終了-------------------------------------")

# reflesh()
