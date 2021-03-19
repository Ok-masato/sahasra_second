import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA

# x-means用
from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.preprocessing import image
from progressbar import ProgressBar


def __feature_extraction(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # resize
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # add a dimention of samples
    x = preprocess_input(x)  # RGB 2 BGR and zero-centering by mean pixel based on the position of channels

    feat = model.predict(x)  # Get image features

    feat = feat.flatten()  # Convert 3-dimentional matrix to (1, n) array

    return feat


np.random.seed(5)
print("Init done.")


# リソースディレクトリの設定、配列追加
unclassified_label = "target"
result_folder = "/img_"
output_path = "."
img_paths = []
for root, dirs, files in os.walk("./" + unclassified_label + "/"):
    for file in files:
        if file.endswith(".jpg"):
            img_paths.append(os.path.join(root, file))
print(img_paths)
img_num = len(img_paths)
print("Image number:", img_num)
print("Image list make done.")

model = ResNet50(weights='imagenet', include_top=False)
# model = VGG16(weights='imagenet', include_top=False)

# model = InceptionResNetV2(weights='imagenet', include_top=False)


X = []
pb = ProgressBar(max_value=len(img_paths))
for i in range(len(img_paths)):
    # Extract image features
    feat = __feature_extraction(model, img_paths[i])
    X.append(feat)
    pb.update(i)  # Update progressbar

# Clutering images by k-means++
X = np.array(X)

# クラスタ数2から探索させてみる
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
# n_clusters = 5
n_clusters = cluster_size
kmeans = KMeans(n_clusters=n_clusters, random_state=5).fit(X)
labels = kmeans.labels_
print("K-means clustering done.")

for i in range(n_clusters):
    label = np.where(labels==i)[0]
    # Image placing
    if not os.path.exists(output_path + result_folder + str(i)):
        os.makedirs(output_path + result_folder + str(i))
    for j in label:
        img = Image.open(img_paths[j])
        fname = img_paths[j].split('/')[-1]
        img.save(output_path + result_folder + str(i)+"/" + fname)
print("Image placing done.")
