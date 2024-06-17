import cv2
from multiprocessing import Process, RawArray, Lock
from detect22 import run2
from detect11 import run1

# 定义摄像头参数
WIDTH = 640
HEIGHT = 640

# 创建共享内存空间
Image_raw = RawArray('B', WIDTH * HEIGHT * 3)
lock = Lock()
# 将摄像头的视频帧存储到共享内存空间
def img2memory(lock, img, raws):
    h, w, _ = img.shape
    lock.acquire()  # 获取线程锁，确保在修改共享内存时只有一个线程访问
    memoryview(raws).cast('B')[:img.size] = img.ravel()  # 将图像数据写入共享内存的对应位置
    lock.release()  # 释放线程锁，允许其他线程访问共享内存
    return w, h  # 返回图像的宽度和高度

#读取摄像头
def read_camera(lock,  raws, fr=0):
        # address = 'rtsp://{}@{}:554/h264/ch1/main/av_stream'  # 创建 RTSP 地址
        #address = 0
        url = "rtsp://admin:qaz123..@192.168.0.123/LiveMedia/ch1/Media1/trackID=1"
        #url = "rtsp://admin:qaz123..@192.168.0.123/LiveMedia/ch1/Media1/trackID=1"
        cap = cv2.VideoCapture(url)  # 打开 RTSP 地址，创建 VideoCapture 对象
        while True:  # 在 flag 为真且时间限制内的循环中读取帧数据
            # 读取帧
            ret, img = cap.read() # 读取一帧图像，将结果存储在 ret, img 变量中
            # 调整帧的大小为共享内存空间的宽度和高度
            img = cv2.resize(img, (WIDTH, HEIGHT))
            if not ret: continue  # 如果没有成功读取到帧，则跳过当前循环
            img2memory(lock, img, raws)  # 调用 img2memory 函数，将帧数据写入共享内存

def main():
    # 创建运行子进程
    process1 = Process(target=run1, args=(Image_raw,))
#    process2 = Process(target=run2, args=(Image_raw,))
    process1.start()
#    process2.start()

    # 创建读取摄像头的线程
    process_main = Process(target=read_camera, args=(lock, Image_raw))
    process_main.start()

    # 主线程等待所有线程结束
    process_main.join()
    process1.join()
#    process2.join()

if __name__ == '__main__':
    main()
