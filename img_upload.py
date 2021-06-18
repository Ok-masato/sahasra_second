import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

"""
画像をFirebase Storageにアップロードする
"""
# キーの設定
cred = credentials.Certificate('./FirebseserviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'firebase-adminsdk-clww9@sahasra-image.iam.gserviceaccount.com'
})
bucket = storage.bucket()


class UseImage:
    # 画像のアップロード
    def image_upload(self, path):
        print(path)


if __name__ == "__main__":

    # アップロード関数にpathを渡す
    folder_path = './Experimental program/target'
    up = UseImage()
    up.image_upload(folder_path)
