import cv2 #引入OpenCV库
import matplotlib.pyplot as plt
import copy #这个就是表明要调用深拷贝模块的命令
import numpy as np
import torch

from src import model
from src import util
from src.body import Body

# 人体姿态检测(身体关节部分)
body_estimation = Body('model/body_pose_model.pth')
print(f"Torch device: {torch.cuda.get_device_name()}")#返回gpu名字，设备索引默认从0开始；

#启动摄像头，3表示width，4表示height
cap = cv2.VideoCapture(0)#捕捉视频
cap.set(3, 640)#宽度设置640
cap.set(4, 480)#高度设置480

#设置视频编码格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 设置视频帧频
fps = cap.get(cv2.CAP_PROP_FPS)
# 设置视频大小
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# VideoWriter方法是cv2库提供的保存视频方法，cv2开源库，用于图像处理、计算机视觉
# 按照设置的格式来out输出，fourcc:用4个字符表示的视频编码格式。 fps:帧速率。 size:每一帧的大小。
out = cv2.VideoWriter('videos/test9.avi',fourcc ,fps, size)

while True:
    ret, oriImg = cap.read()#读取每一帧图像，第一个参数ret的值为True或False，代表有没有读到图片
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)#深copy这个图像到画框
    canvas = util.draw_bodypose(canvas, candidate, subset)#绘制身体关节点

    if ret == True:
        # 垂直翻转矩阵
        # frame = cv2.flip(frame,0)
        out.write(canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):#按下q键后break
            break
    else:
        break

# 释放资源
cap.release()
out.release()
# 关闭窗口
cv2.destroyAllWindows()