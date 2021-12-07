import os

import cv2

IMAGE_DIR = 'images'

class ImageOnModel():
    def __init__(self, model_id: int, model_name: str) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.image_path = os.path.join(IMAGE_DIR, model_name + ".png")
        self.image = cv2.imread(self.image_path, -1)


class ObjectWithImage():
    def __init__(self, object, iom_data: ImageOnModel) -> None:
        self.object = object   #画像を重ねるオブジェクト
        self.iom = iom_data #対応するImageOnModelクラスのインスタンス
        self.width = self.object[2] - self.object[0]
        self.height = self.object[3] - self.object[1]
        self.confidence = self.object[4]
        self.x = self.object[0] + self.width * 0.5
        self.y = self.object[1] + self.height * 0.5

    def mix_image(self, back, mag):
        front4 = self.iom.image
        fh, fw = front4.shape[:2]
        bh, bw = back.shape[:2]
        if not ((-fw/2 < self.x < bw+(fw/2)) and (-fh/2 < self.y < bh+(fh/2))):
            return back

        if self.width > self.height:
            fh = int(fh * (self.width / fw) * mag)
            fw = int(self.width * mag)
        else:
            fw = int(fw * (self.height / fh) * mag)
            fh = int(self.height * mag)

        front4 = cv2.resize(front4, dsize=(fw, fh))
        fh, fw = front4.shape[:2]
        fx_min = int(self.x-fw/2)
        fy_min = int(self.y-fh/2)
        fx_max = fx_min + fw
        fy_max = fy_min + fh
        x1, y1 = max(fx_min, 0), max(fy_min, 0)
        x2, y2 = min(fx_max, bw), min(fy_max, bh)
        front3 = front4[:, :, :3]
        mask1 = front4[:, :, 3:]
        mask3 = 255 - cv2.merge((mask1, mask1, mask1))
        mask_roi = mask3[y1-fy_min:y2-fy_min, x1-fx_min:x2-fx_min]
        front_roi = front3[y1-fy_min:y2-fy_min, x1-fx_min:x2-fx_min]
        roi = back[y1:y2, x1:x2]
        tmp = cv2.bitwise_and(roi, mask_roi)
        tmp = cv2.bitwise_or(tmp, front_roi)
        back[y1:y2, x1:x2] = tmp
        return back

    def print_object(self):
        print(f"種類id:{(int)(self.object[5]): >2}", end=", ")
        print(f"種類name:{self.iom.model_name: >10}", end=", ")
        print(f"xmin:{(int)(self.object[0]): >4}", end=", ")
        print(f"xmax:{(int)(self.object[2]): >4}", end=", ")
        print(f"ymin:{(int)(self.object[1]): >4}", end=", ")
        print(f"ymax:{(int)(self.object[3]): >4}", end=", ")
        print(f"信頼度:{(self.object[4]*100):.1f}%", end=", ")
        print(f"幅:{self.width: >3}", end=", ")
        print(f"高さ:{self.height: >3}")
