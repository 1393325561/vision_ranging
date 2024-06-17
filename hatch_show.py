import time
import threading
import sys
import cv2
import torch
from os import getcwd
import argparse    #这个库用于解析命令行参数。
import numpy as np
import random
from hatch_ui import Ui_MainWindow
from PySide6.QtWidgets import QApplication,QMainWindow,QMessageBox,QFileDialog
from PySide6 import QtCore,QtWidgets,QtGui
from models.experimental import attempt_load
from qt_material import apply_stylesheet
from PySide6.QtGui import QPixmap,QImage
from PySide6.QtCore import Qt
from utils.augmentations import letterbox
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device, time_sync

class Mywindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super().__init__()

        self.setupUi(self)

        self.path = getcwd()  # 使用 os 模块中的 getcwd() 函数设置为当前工作目录
        self.setWindowTitle('机场异物检测系统')
        self.vid_source = 0 # 初始设置为摄像头
        self.stopEvent = threading.Event()
        self.model_load()
        self.det_image = None

        self.cap_video = None
        self.cap = None
        self.timer_video = QtCore.QTimer()
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture(self.vid_source)  # 屏幕画面对象
        #self.model = self.model_load(weights="yolov5s.pt",device=self.device)
        self.count_table = []
        self.bind()

    def bind(self):
        self.btn_in_img.clicked.connect(self.in_img)
        self.btn_det_img.clicked.connect(self.det_img)

        self.btn_save_img.clicked.connect(self.save_image)

        self.btn_in_camera.clicked.connect(self.show_camera)
        self.timer_camera.timeout.connect(self.in_camera)
        self.timer_camera.timeout.connect(self.det_camera)
        self.btn_close_camera.clicked.connect(self.closeEvent)

        self.btn_in_video.clicked.connect(self.show_video)
        self.timer_video.timeout.connect(self.det_video)
        self.btn_cho.clicked.connect(self.choose_model)
        #self.btn_in_video.clicked.connect(self.button_open_camera_click)


    def model_load(self,model_path = None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt',
                            help='model.pt path(s)')  # 模型路径仅支持.pt文件
        parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')  # 检测图像大小，仅支持480
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')  # 置信度阈值
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')  # NMS阈值
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--save-dir', type=str, default='inference', help='directory to save results')  # 文件保存路径
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')  # 分开类别
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')  # 使用NMS
        self.opt = parser.parse_args()  # opt局部变量，重要
        weight, imgsz = self.opt.weights, self.opt.img_size
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # 如果使用gpu则进行半精度推理
        if model_path:
            weight = model_path
        self.model = attempt_load(weight, map_location=self.device)  # 读取模型
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # 检查图像尺寸
        if self.half:  # 如果是半精度推理
            self.model.half()  # 转换模型的格式
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # 得到模型训练的类别名

        color = [[132, 56, 255], [82, 0, 133], [203, 56, 255], [255, 149, 200], [255, 55, 199],
                 [72, 249, 10], [146, 204, 23], [61, 219, 134], [26, 147, 52], [0, 212, 187],
                 [255, 56, 56], [255, 157, 151], [255, 112, 31], [255, 178, 29], [207, 210, 49],
                 [44, 153, 168], [0, 194, 255], [52, 69, 147], [100, 115, 255], [0, 24, 236]]
        self.colors = color if len(self.names) <= len(color) else [[random.randint(0, 255) for _ in range(3)] for _ in
                                                                   range(len(self.names))]  # 给每个类别一个颜色
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # 创建一个图像进行预推理
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # 预推理
        print('模型加载完成！')

    def cv_imread(self,filePath):
        # 读取图片
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)

        if len(cv_img.shape) > 2:
            if cv_img.shape[2] > 3:
                cv_img = cv_img[:, :, :3]
        return cv_img
    def det_img(self):
        self.timer_camera.stop()
        self.timer_video.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        #self.clearUI()
        if self.path == "":

            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            self.lineEdit_show.setText(self.path + '文件已选中')
            self.label_display.setText('正在检测，请稍等！.....')
            image = self.cv_imread(self.path)
            image = cv2.resize(image, (640, 640))
            img0 = image.copy()
            img = letterbox(img0, new_shape=self.imgsz)[0]
            img = np.stack(img, 0)
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)  # 转化到GPU上
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # 增加一个维度
            t1 = time_sync()
            pred = self.model(img, augment=False)[0]  # 前向推理
            """
               pred.shape=(1, num_boxes, 5+num_class)
               h,w为传入网络图片的长和宽,注意dataset在检测时使用了矩形推理,所以这里h不一定等于w
               num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
               pred[..., 0:4]为预测框坐标=预测框坐标为xywh(中心点+宽长)格式
               pred[..., 4]为objectness置信度
               pred[..., 5:-1]为分类结果
               """

            # NMS
            """
            pred: 网络的输出结果
            conf_thres:置信度阈值
            ou_thres:iou阈值
            classes: 是否只保留特定的类别
            agnostic_nms: 进行nms是否也去除不同类别之间的框
            max-det: 保留的最大检测框数量
            ---NMS, 预测框格式: xywh(中心点+长宽)-->xyxy(左上角右下角)
            pred是一个列表list[torch.tensor], 长度为batch_size
            每一个torch.tensor的shape为(num_boxes, 6), 内容为box + conf + cls
            """
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,agnostic=self.opt.agnostic_nms)  # NMS过滤
            t2 = time_sync()
            InferenceNms = t2 - t1
            print(InferenceNms)
            self.label_show_time.setText(str(round(InferenceNms, 4)))
            det = pred[0]
            # p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
            # s: 输出信息 初始为 ''
            # im0: 原始图片 letterbox + pad 之前的图片
            p, s, im0 = None, '', img0

            if det is not None and len(det):  # 如果有检测信息则进入
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # 把图像缩放至im0的尺寸
                number_i = 0  # 类别预编号
                self.detInfo = []
                #count = [0 for i in self.count_name]
                for *xyxy, conf, cls in reversed(det):

                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    # 将检测信息添加到字典中
                    self.detInfo.append(
                        [self.names[int(cls)], [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf, int(cls)])
                    number_i += 1  # 编号数+1
                    print(self.names[int(cls)])
                    #self.label_show_class.setText(str(self.names[int(cls)]))   # 显示类别
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    im0 = self.plot_one_box(image, xyxy, color=self.colors[int(cls)], label=label)
                    # for _ in range(len(det)):
                    #     self.count_table.append(count)  # 记录各个类别数目
                    self.det_image = im0
                    pixmap = self.cv_change(im0)
                    # 设置 QPixmap 的尺寸以适应 QLabel 的大小，并保持纵横比
                    self.label_show_class.setText(str(number_i))
                    self.label_out_img.setPixmap(pixmap.scaled(self.label_out_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def plot_one_box(self,img, x, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img
    def in_img(self):
            # 选择文件对话框，用于选择图片文件
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', self.path, '*.jpg *.png *.tif *.jpeg')
              #fileName为  D:/yolo/yolov5-mask-42-master/2.JPG
        self.path = fileName  # 保存原始图片路径，可能用于后续预测
        if fileName:  # 如果选择了文件
            # 读取图片，并按比例调整图片大小
            im0 = cv2.imread(self.path)  # 读取图片
            pixmap=self.cv_change(im0)
            # 将调整后的图片显示在左侧的图像显示控件中
            # 设置 QPixmap 的尺寸以适应 QLabel 的大小，并保持纵横比
            self.label_in_img.setPixmap(pixmap.scaled(self.label_in_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            #self.label_out_img.setPixmap(QPixmap("images/UI/right.jpeg"))  # 将右侧的图像显示控件设置为默认图片

    def cv_change(self,im0):
        img_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        # 将OpenCV的图像转换为QImage
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 将QImage转换为QPixmap用于显示
        pixmap = QPixmap.fromImage(qimg)
        return pixmap

    def show_video(self):
        if self.timer_camera.isActive():
            self.timer_camera.stop()
        QtWidgets.QApplication.processEvents()
        if self.cap:
            self.cap.release()  # 释放视频画面帧
        QtWidgets.QApplication.processEvents()

        if not self.timer_video.isActive():  # 检查定时状态
            fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose video file', self.path, '*.mp4 *.avi')
            if fileName:
                self.lineEdit_show2.setText(self.path + '文件已选中')
                self.label_display_2.setText('正在启动识别系统...\n\nleading')
                QtWidgets.QApplication.processEvents()

                try:  # 初始化视频流
                    self.cap_video = cv2.VideoCapture(fileName)
                except:
                    print("[INFO] could not determine # of frames in video")
                self.timer_video.start(30)  # 打开定时器
        else:
            # 定时器未开启，界面回复初始状态
            self.timer_video.stop()
            self.cap_video.release()
            self.label_display_2.clear()
            time.sleep(0.5)#程序将会暂停执行0.5秒钟

            QtWidgets.QApplication.processEvents()

    def show_camera(self):
        self.vid_source=0
        if self.timer_video.isActive():
            self.timer_video.stop()
        QtWidgets.QApplication.processEvents()
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧


        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.vid_source)    # 检查相机状态
            if flag == False:        # 相机打开失败提示
                msg = QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"请检测相机与电脑是否连接正确",
                    buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:

                self.lineEdit_show2.setText('实时摄像已启动')
                self.label_display_2.setText('正在启动视频识别...\n\nleading')
                QtWidgets.QApplication.processEvents()
                self.timer_camera.start(30)
        else:
            # 定时器未开启，界面回复初始状态
            self.timer_camera.stop()
            if self.cap:
                self.cap.release()
            QtWidgets.QApplication.processEvents()

    def det_camera(self):
        # 定时器槽函数，每隔一段时间执行
        flag, image = self.cap.read()
        if flag:
            image = cv2.resize(image, (850, 500))
            img0 = image.copy()
            img = letterbox(img0, new_shape=self.imgsz)[0]
            img = np.stack(img, 0)
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)  # 转化到GPU上
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # 增加一个维度
            t1 = time_sync()
            pred = self.model(img, augment=False)[0]  # 前向推理

            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)  # NMS过滤
            t2 = time_sync()
            InferenceNms = t2 - t1
            print(InferenceNms)
            self.label_show_time.setText(str(round(InferenceNms, 4)))
            det = pred[0]
            # p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
            # s: 输出信息 初始为 ''
            # im0: 原始图片 letterbox + pad 之前的图片
            p, s, im0 = None, '', img0

            if det is not None and len(det):  # 如果有检测信息则进入
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # 把图像缩放至im0的尺寸
                number_i = 0  # 类别预编号
                self.detInfo = []

                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    # 将检测信息添加到字典中
                    self.detInfo.append(
                        [self.names[int(cls)], [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf, int(cls)])
                    number_i += 1  # 编号数+1
                    # label = '%s %.0f%%' % (self.names[int(cls)], conf * 100)
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    im0 = self.plot_one_box(image, xyxy, color=self.colors[int(cls)], label=label)

                    self.det_image = im0
                    pixmap = self.cv_change(im0)
                    # 将调整后的图片显示在左侧的图像显示控件中
                    # 设置 QPixmap 的尺寸以适应 QLabel 的大小，并保持纵横比
                    self.label_outvideo.setPixmap(pixmap.scaled(self.label_outvideo.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def det_video(self):
        # 定时器槽函数，每隔一段时间执行

        flag, image = self.cap_video.read()
        if flag:
            image = cv2.resize(image, (640, 640))
            img0 = image.copy()
            img = letterbox(img0, new_shape=self.imgsz)[0]
            img = np.stack(img, 0)
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)  # 转化到GPU上
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # 增加一个维度
            t1 = time_sync()
            pred = self.model(img, augment=False)[0]  # 前向推理

            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)  # NMS过滤
            t2 = time_sync()
            InferenceNms = t2 - t1
            print(InferenceNms)
            self.label_show_time.setText(str(round(InferenceNms, 4)))
            det = pred[0]
            # p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
            # s: 输出信息 初始为 ''
            # im0: 原始图片 letterbox + pad 之前的图片
            p, s, im0 = None, '', img0

            if det is not None and len(det):  # 如果有检测信息则进入
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()  # 把图像缩放至im0的尺寸
                number_i = 0  # 类别预编号
                self.detInfo = []

                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    # 将检测信息添加到字典中
                    self.detInfo.append(
                        [self.names[int(cls)], [c1[0], c1[1], c2[0], c2[1]], '%.2f' % conf, int(cls)])
                    number_i += 1  # 编号数+1
                    # label = '%s %.0f%%' % (self.names[int(cls)], conf * 100)
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    im0 = self.plot_one_box(image, xyxy, color=self.colors[int(cls)], label=label)

                    self.det_image = im0
                    pixmap = self.cv_change(im0)
                    # 将调整后的图片显示在左侧的图像显示控件中
                    # 设置 QPixmap 的尺寸以适应 QLabel 的大小，并保持纵横比
                    self.label_outvideo.setPixmap(pixmap.scaled(self.label_outvideo.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    def in_camera(self):
        flag, self.image = self.cap.read()

        self.image=cv2.flip(self.image, 1) # 左右翻转
        show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        showImage = QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(showImage)
        self.label_invideo.setPixmap(pixmap.scaled(self.label_outvideo.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.label_invideo.setScaledContents(True)


    def closeEvent(self):
        if self.timer_camera.isActive():
            # 创建按钮
            ok = QtWidgets.QPushButton('确定')
            cancel = QtWidgets.QPushButton('取消')

            # 创建并设置消息框
            msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, "关闭", "是否关闭！")
            msg.addButton(ok, QtWidgets.QMessageBox.AcceptRole)
            msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
            ok.setText(u'确定')
            cancel.setText(u'取消')
            # 显示消息框并等待用户响应
            if msg.exec() != QtWidgets.QMessageBox.RejectRole:
                # 如果用户点击“取消”，则忽略关闭事件
            # 用户选择关闭窗口，停止摄像头并释放资源
                if self.cap.isOpened():
                    self.cap.release()
                if self.timer_camera.isActive():
                    self.timer_camera.stop()


    def choose_model(self):
        self.timer_camera.stop()
        self.timer_video.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧

        # self.comboBox_select.clear()  # 下拉选框的显示
        # self.comboBox_select.addItem('所有目标')  # 清除下拉选框


        # 调用文件选择对话框
        fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget,
                                                                "选取图片文件", getcwd(),  # 起始路径
                                                                "Model File (*.pt)")  # 文件类型
        # 显示提示信息
        if fileName_choose != '':
            self.btn_cho.setToolTip(fileName_choose + ' 已选中')
        else:
            fileName_choose = None  # 模型默认路径
            self.btn_cho.setToolTip('使用默认模型')
        self.model_load(fileName_choose)
    def save_image(self):
        if self.det_image is not None:
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            cv2.imwrite('./pic_' + str(now_time) + '.png', self.det_image)
            QMessageBox.about(self.centralwidget, "保存文件", "\nSuccessed!\n文件已保存！")
        else:
            QMessageBox.about(self.centralwidget, "保存文件", "saving...\nFailed!\n请先选择检测操作！")
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Mywindow()
    # app = QtWidgets.QApplication(sys.argv)
    # window = QtWidgets.QMainWindow()
    #apply_stylesheet(app, theme='light_purple.xml')
    window.show()
    app.exec()