import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# x-means用
from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
# deep leaning特徴量抽出用
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.preprocessing import image
from progressbar import ProgressBar


class AutomaticClassification:
    def __init__(self, target, output_path, result_folder, model_name):
        self.target = target
        self.output_path = output_path
        self.result_folder = result_folder
        self.model_name = model_name
        self.img_paths = []
        self.img_num = 0

        # 使用するモデルのインポート
        if self.model_name == "resnet50":
            self.model = ResNet50(weights='imagenet', include_top=False)
        elif self.model_name == "vgg16":
            self.model = VGG16(weights='imagenet', include_top=False)
        elif self.model_name == "inception_resnet_v2":
            self.model = InceptionResNetV2(weights='imagenet', include_top=False)
        else:
            print("モデル名が間違っています")

    def update_paramater(self, target, output_path, result_folder):
        self.target = target
        self.output_path = output_path
        self.result_folder = result_folder
        self.img_paths = []
        self.img_num = 0

    def import_img(self):
        np.random.seed(5)
        for root, dirs, files in os.walk(self.target):
            for file in files:
                if file.endswith(".jpg"):
                    self.img_paths.append(os.path.join(root, file))
        print(self.img_paths)
        self.img_num = len(self.img_paths)
        print("Image number:", self.img_num)
        print("Image list make done.")

    def start_classification(self):
        X = []
        pb = ProgressBar(max_value=len(self.img_paths))
        for i in range(len(self.img_paths)):
            # Extract image features
            feat = self.__feature_extraction(self.img_paths[i])
            X.append(feat)
            pb.update(i)

        # Clutering images by k-means++
        X = np.array(X)
        self.x_means(X)

    def __feature_extraction(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))  # resize
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # add a dimention of samples
        x = preprocess_input(x)  # RGB 2 BGR and zero-centering by mean pixel based on the position of channels
        feat = self.model.predict(x)  # Get image features
        feat = feat.flatten()  # Convert 3-dimentional matrix to (1, n) array
        return feat

    def x_means(self, X):
        try:
            # クラスタ数=2から探索させてみる
            initial_centers = kmeans_plusplus_initializer(X, 2).initialize()
            # クラスタリングの実行
            instances = xmeans(X, initial_centers, ccore=True)
            instances.process()
            # クラスタはget_clustersで取得できる
            clusters = instances.get_clusters()
            # 最適クラスタサイズを取得
            cluster_size = len(clusters)
            print("\n Cluster Size: " + str(cluster_size))

            # K-means によるクラスタリング
            n_clusters = cluster_size
            kmeans = KMeans(n_clusters=n_clusters, random_state=5).fit(X)
            labels = kmeans.labels_
            print("K-means clustering done.")

            for i in range(n_clusters):
                label = np.where(labels == i)[0]
                # Image placing
                if not os.path.exists(self.output_path + self.result_folder + str(i)):
                    os.makedirs(self.output_path + self.result_folder + str(i))
                for j in label:
                    img = Image.open(self.img_paths[j])
                    fname = os.path.basename(self.img_paths[j])
                    img.save(self.output_path + self.result_folder + str(i) + "/" + fname)
            print("Image placing done.")
        # 分類する画像枚数が初期クラスタ数より少ない場合ValueErrorがでるので回避する
        except ValueError:
            print("分類に必要な画像の枚数に達していなかったのでスキップします")
