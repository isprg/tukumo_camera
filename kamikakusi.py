import torch
import cv2

# names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#         'hair drier', 'toothbrush']  # class names


def print_object(xmin,ymin,xmax,ymax,confidence,class_id):
    width = xmax - xmin
    height = ymax - ymin
    print(f"種類:{class_id: >2}", end=", ")
    print(f"xmin:{xmin: >4}", end=", ")
    print(f"xmax:{xmax: >4}", end=", ")
    print(f"ymin:{ymin: >4}", end=", ")
    print(f"ymax:{ymax: >4}", end=", ")
    print(f"信頼度:{confidence:.3f}", end=", ")
    print(f"幅:{width: >3}", end=", ")
    print(f"高さ:{height: >3}")

def mix_png(back, front4, center_pos, model_width, model_height):
    x, y = center_pos    
    fh, fw = front4.shape[:2]
    bh, bw = back.shape[:2]
    if not ((-fw/2 < x < bw+(fw/2)) and (-fh/2 < y < bh+(fh/2))) :
        return back

    if model_width > model_height:
        fh = int(fh * (model_width / fw))
        fw = model_width
    else:
        fw = int(fw * (model_height / fh))
        fh = model_height

    front4 = cv2.resize(front4, dsize=(fw, fh))
    fh, fw = front4.shape[:2]
    fx_min = int(x-fw/2)
    fy_min = int(y-fh/2)
    fx_max = fx_min + fw
    fy_max = fy_min + fh
    x1, y1 = max(fx_min, 0), max(fy_min, 0)
    x2, y2 = min(fx_max, bw), min(fy_max, bh)
    front3 = front4[:, :, :3]
    mask1 = front4[:, :, 3:]
    mask3 = 255 - cv2.merge((mask1, mask1, mask1))
    # print(front3.shape)
    # print(mask3.shape)
    mask_roi = mask3[y1-fy_min:y2-fy_min, x1-fx_min:x2-fx_min]
    front_roi = front3[y1-fy_min:y2-fy_min, x1-fx_min:x2-fx_min]
    roi = back[y1:y2, x1:x2]
    # print(mask_roi.shape)
    # print(front_roi.shape)
    # print(roi.shape)
    tmp = cv2.bitwise_and(roi, mask_roi)
    tmp = cv2.bitwise_or(tmp, front_roi)
    back[y1:y2, x1:x2] = tmp
    return back

def putSprite_mask(back, front4, pos):
    x, y = pos
    fh, fw = front4.shape[:2]
    bh, bw = back.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x+fw, bw), min(y+fh, bh)
    if not ((-fw < x < bw) and (-fh < y < bh)) :
        return back
    front3 = front4[:, :, :3]
    mask1 = front4[:, :, 3]
    mask3 = 255 - cv2.merge((mask1, mask1, mask1))
    mask_roi = mask3[y1-y:y2-y, x1-x:x2-x]
    front_roi = front3[y1-y:y2-y, x1-x:x2-x]
    roi = back[y1:y2, x1:x2]
    tmp = cv2.bitwise_and(roi, mask_roi)
    tmp = cv2.bitwise_or(tmp, front_roi)
    back[y1:y2, x1:x2] = tmp
    return back

def trance_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

def main():
    # デバイス等の定数設定
    camera_id = 0
    delay = 1
    model_filter = ['cell phone']
    back_image = cv2.imread('WIN_20211110_15_07_16_Pro.jpg')
 
    # モデルの読み込み
    model = torch.hub.load("ultralytics/yolov5", "yolov5s6", pretrained=True)
    # model = torch.load("yolov5m.pt") # オフライン時の暫定モデルファイルの読み込み
    # print(model.names)  # 検出できる物体の種類

    # if not model_filter:
        # model_filter = model.names
    
    # キャプチャの初期値
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 24)
    ret, frame = cap.read()

    while cap.isOpened():
        if ret:
            results = model(frame)  # 画像パスを設定し、物体検出を行う
            objects = results.xyxy[0].cpu().numpy()  # 検出結果を取得
            # print(objects)

            for object in objects:
                class_id = int(object[5])
                if model.names[class_id] in model_filter:
                    xmin = int(object[0])
                    ymin = int(object[1])
                    xmax = int(object[2])
                    ymax = int(object[3])
                    confidence = object[4]
                    crop_img = frame[ymin:ymax, xmin:xmax, :]

            # cv2.imshow("source", frame)  # 検出した物体の表示
            cv2.imshow("mixed", mixed)
            # results.crop()  # 検出した物体の切り取り

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        
        ret, frame = cap.read()

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()