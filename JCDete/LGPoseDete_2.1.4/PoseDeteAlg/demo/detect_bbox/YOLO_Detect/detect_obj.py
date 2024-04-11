# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:52:08 2020

@author: huqiugen
"""
import argparse
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import time
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-1cls.cfg', help='*.cfg path')
parser.add_argument('--weights', type=str, default='weights/last.pt', help='path to weights file')
parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
parser.add_argument('--conf_thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou_thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
opt = parser.parse_args()
img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
weights = opt.weights
# 初始化
device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
# 初始化模型
model = Darknet(opt.cfg, img_size)
# 加载权重
attempt_download(weights)
if weights.endswith('.pt'):  # pytorch格式
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
else:  # darknet格式
    _ = load_darknet_weights(model, weights)
# Eval模式
model.to(device).eval()
def detect(img0):
    dataset = LoadImages(img0, img_size=img_size, half=False)
    # 运行推测,得到检测
    img = torch.from_numpy(dataset).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  
    pred = model(img)[0]
    # NMS应用
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None)
    for i, det in enumerate(pred):  # 检测每一个图像
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            result = det.cpu().detach().numpy()
            return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-1cls.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='path to weights file')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--display', default=True, help='Display online ')
    parser.add_argument('--input_img', type=str, default='data/samples', help='input image list')
    parser.add_argument('--save_name', type=str, default='temp', help='output save image')
    opt = parser.parse_args()
    display = opt.display
    input_img = opt.input_img
    save_name = opt.save_name
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    with torch.no_grad():
        path = str(Path(input_img))
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]
        for i in files:
            filepath, tempfilename = os.path.split(i)
            img0 = cv2.imread(i)
            t1 = time.time()
            result = detect(img0)
            bbox = result[:, :4]
            print('bbox', bbox)
            bbox_center = []
            for m, bb in enumerate(bbox):
                xmin, ymin, xmax, ymax = bbox[m]
                x = (xmax + xmin) / 2
                y = (ymax + ymin) / 2
                bbox_center.append([x, y])
                cv2.rectangle(img0, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1)
                cv2.circle(img0, (int(bbox_center[m][0]), int(bbox_center[m][1])), 2, (0, 255, 0), 4)
            if display:
                cv2.imwrite(save_name + '/' + tempfilename, img0)
            t2 = time.time()
            t3 = t2-t1
            print('t3:',t3)
