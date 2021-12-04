import os

import torch
import cv2

from common import ImageOnModel, ObjectWithImage

IMAGE_DIR = 'images'
CAMERA_ID = 0
DELAY = 1

def read_iom(model):
    iom_datas = {}
    for i, name in enumerate(model.names):
        image_path = os.path.join(IMAGE_DIR, name + '.png')
        if os.path.isfile(image_path):
            iom_datas[i] = ImageOnModel(i, name)
    return iom_datas

def main():
    # モデルの読み込み
    model = torch.hub.load("ultralytics/yolov5", "yolov5s6", pretrained=True)
    # model = torch.hub.load('path/to/yolov5', 'custom', path='path/to/best.pt', source='local')  # ローカルのモデル
    # print(model.names)  # 検出できる物体の種類

    # モデルと画像を対応付け
    iom_datas = read_iom(model)

    # キャプチャの初期値
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ret, frame = cap.read()

    #動画処理 & 動画出力
    while cap.isOpened():
        if ret:
            frame_mixed = frame.copy()
            frame_mixed = cv2.flip(frame_mixed, 1)
            results = model(frame_mixed)  # 画像パスを設定し、物体検出を行う
            objects = results.xyxy[0].numpy()  # 検出結果を取得
            # print(objects)

            for object in objects:
                model_id = int(object[5])
                if model_id in iom_datas:
                    owi_data = ObjectWithImage(object, iom_datas[model_id])
                    owi_data.print_object()
                    frame_mixed = owi_data.mix_image(frame_mixed, 1.0)

            # cv2.imshow("source", frame)  # 処理前の映像表示
            cv2.imshow("mixed_iamge", frame_mixed)  # 処理後の映像表示

        if cv2.waitKey(DELAY) & 0xFF == ord('q'):
            break
        
        ret, frame = cap.read()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()