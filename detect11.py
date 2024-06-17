import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.augmentations import letterbox
from utils.torch_utils import select_device

def run1(image_path):
    img_size = 640
    stride = 32
    weights = 'yolov5s.pt'  # 模型权重文件路径
    device = 'cpu'  # 设置设备类型
    # image_path = 'data/images/bus.jpg'  # 输入图像路径（也可以是绝对路径）
    save_path = 'run1/11.jpg'  # 输出图像保存路径（也可以是绝对路径）
    view_img = True  # 是否显示检测结果的图像窗口
    half = False

    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # 导入模型
    model = attempt_load(weights, map_location=device)  # 加载模型
    img_size = check_img_size(img_size, s=stride)  # 检查图像尺寸是否符合规范
    names = model.names  # 获取模型输出的类别名称（people、bus等）

    while True:
        # Padded resize
        # img0 = cv2.imread(image_path)  # 读取输入图像
        img0 = np.frombuffer(image_path, dtype=np.uint8).reshape((640, 640, 3))

        img = letterbox(img0, img_size, stride=stride, auto=True)[0]  # 对输入图像进行填充和调整大小

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # 图像通道转换和颜色通道转换
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0   # 归一化图像数据
        img = img[None]     # [h w c] -> [1 h w c]

        # inference
        pred = model(img)[0]  # 进行目标检测
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000)  # 非最大抑制，保留置信度最高的目标框

        # plot label
        det = pred[0]
        annotator = Annotator(img0, line_width=3, example=str(names))  # 创建绘制标签的对象
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()  # 将检测结果的坐标转换到原始图像坐标系中
            for *xyxy, conf, cls in reversed(det):  # 遍历每个目标框
                c = int(cls)  # 目标类别
                label = f'{names[c]} {conf:.2f}'  # 标签文字内容
                annotator.box_label(xyxy, label, color=colors(c, True))  # 绘制标签框和文字

        # write image
        im0 = annotator.result()  # 获取带有标签的图像
        cv2.imwrite(save_path, im0)  # 保存图像
        if view_img:
            im0 = cv2.resize(im0, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC) # 按比例修改需要展示的图像大小
            cv2.imshow(str('image-1'), im0)  # 显示图像
            cv2.waitKey(1)
        print(f'Inference {image_path} finish, save to {save_path}') # 打印图片来源和输出保存路径
