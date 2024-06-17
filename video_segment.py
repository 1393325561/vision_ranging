# import os
# import cv2
# # 定义保存图片函数
# # image:要保存的图片
# # pic_address：图片保存地址
# # num: 图片后缀名，用于区分图片，int 类型
# def save_image(image, address, num):
#     pic_address = address + str(num) + '.jpg'
#     cv2.imwrite(pic_address, image)
#
# def video_to_pic(video_path, save_path, frame_rate):  # 读取视频文件
#     videoCapture = cv2.VideoCapture(video_path)
#     j = 0
#     i = 0
#     # 读帧
#     success, frame = videoCapture.read()
#     while success:
#         i = i + 1
#         # 每隔固定帧保存一张图片
#         if i % frame_rate == 0:
#             j = j + 1
#             save_image(frame, save_path, j)
#             print('图片保存地址：', save_path + str(j) + '.jpg')
#         success, frame = videoCapture.read()
# if __name__ == '__main__':  # 视频文件和图片保存地址
#
#     SAMPLE_VIDEO = './Airport hatch/'
#     SAVE_PATH = './images2/'
#
#     if not os.path.exists(SAVE_PATH):
#         os.makedirs(SAVE_PATH)
#     # 设置固定帧率
#     FRAME_RATE = 50
#     video_to_pic(SAMPLE_VIDEO, SAVE_PATH, FRAME_RATE)
import cv2
import os

'''
定义保存图片的函数
image：要保存的图片
addr：图片的地址和名称信息
num图片名称的后缀，使用int类型来计数
'''


def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)


def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)


def video2img(filename, timeF):
    #   读取视频文件
    vode = cv2.VideoCapture("./Airport hatch/" + filename)
    #   读帧
    success, frame = vode.read()
    #   初始化变量
    i = 0  # 帧计数
    j = 0  # 图片计数
    timeF = timeF  # 每隔57帧（一秒）保存一张图片，这个要看自己的视频每秒是多少帧

    output_path = "./image/" + filename  # 创建文件夹
    setDir(output_path)

    # 使用循环进行图片的保存
    while success:
        i = i + 1
        if (i % timeF == 0):
            j = i + 1
            save_image(frame, './image/' + filename+'/image_' , j)
            print(filename + ':save image:', j)
        success, frame = vode.read()


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.mp4':
            list_name.append(file_path.split('\\')[1])


if __name__ == '__main__':
    timeF = 25  # 每隔多少帧（一秒）保存一张图片，这个要看自己的视频每秒是多少帧

    list_name = []
    path = "./Airport hatch"
    listdir(path, list_name)

    for filename in list_name:
        video2img(filename, timeF)
        print()