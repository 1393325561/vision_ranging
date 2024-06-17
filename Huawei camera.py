import cv2
import numpy as np
url = "rtsp://admin:qaz123..@192.168.0.120/LiveMedia/ch1/Media1/trackID=1"#=1实况流，=4元数据
cap = cv2.VideoCapture(url)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置图像宽度
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 设置图像高度
ret, frame = cap.read()#参数ret 为True 或者False,代表有没有读取到图片，第二个参数frame表示截取到一帧的图片


while ret:
    #获取设备参数，cv.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT是cv2提供的参数选项
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # 我这里是1280.0 720.0
    print(width, height)
    ret, frame = cap.read()
    cv2.imshow("frame",frame)
    img1 = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('img', img1)
    res = np.hstack((frame, img1))
    cv2.imshow('img', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#    if cv2.getWindowProperty('lyh', cv2.WND_PROP_AUTOSIZE) < 1:  # 用鼠标点击窗口退出键实现退出循环
#        break
cv2.destroyAllWindows()
cap.release()
# import cv2
# def run_opencv_camera():
#     #video_stream_path = 0  # local camera (e.g. the front camera of laptop)
#     video_stream_path = "rtsp://admin:qaz123..@192.168.0.123/LiveMedia/ch1/Media1"
#     cap = cv2.VideoCapture(video_stream_path)
#
#     while cap.isOpened():
#         is_opened, frame = cap.read()
#         cv2.imshow('frame', frame)
#         cv2.waitKey(1)
#     cap.release()
# run_opencv_camera()

#
# import cv2
# import numpy as np
# url = "rtsp://admin:qaz123..@192.168.0.123/LiveMedia/ch1/Media1/trackID=1"  # =1实况流，=4元数据
# import time
# import cv2
# import numpy as np
# import torch
#
# def detect():
#      cap = cv2.VideoCapture(url)
#      #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 设置图像高度
#
#      while (True):
#        #cap = cv2.VideoCapture(url)
#        x=cap.get(cv2.CAP_PROP_FPS)
#        cap.set(cv2.CAP_PROP_FPS,10)
#        y=cap.get(cv2.CAP_PROP_FPS)
#        #print(x)
#        print(y)
#        ret, im1 = cap.read()
#
#        cv2.imshow("111", im1)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#          break
#      #cv2.destroyAllWindows()
#      #cap.release()
# if __name__ == "__main__":
#     detect()
# # import cv2
# # url = "rtsp://admin:qaz123..@192.168.0.123/LiveMedia/ch1/Media1/trackID=1"  # =1实况流，=4元数据
# # cap = cv2.VideoCapture(url)
# #
# # while (True):
# #   ret, im1 = cap.read()
# #   cv2.imshow("111", im1)
# #   if cv2.waitKey(1) & 0xFF == ord('q'):
# #      break
# # cv2.destroyAllWindows()
# # cap.release()