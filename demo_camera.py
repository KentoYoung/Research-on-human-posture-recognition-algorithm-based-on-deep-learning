import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

from src import model
from src import util
from src.body import Body


body_estimation = Body('model/body_pose_model.pth')


print(f"Torch device: {torch.cuda.get_device_name()}")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while True:
    ret, oriImg = cap.read()
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    cv2.imshow('demo', canvas)#一个窗口用以显示原视频
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

