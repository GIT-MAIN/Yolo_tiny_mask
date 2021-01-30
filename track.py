# coding: utf8

"""
搭建环境：tensorflow-gpu2.0.9+keras2.0+opencv3

训练集库：MIT+ISP行人运动图片集

训练网络：基于keras的序列模型cnn

//////////////////////////////////////////
主程序步骤：
             加载视频
                |
            加载已经训练好的模型
                |
            加载背景分割模型：MOG2并训练
                |
            使用背景分割器计算前景掩码，并进行二值化处理
                |
            形体学膨胀检测目标轮廓
                |
            利用训练模型分拣器提取目标roi值，检测两个框的重叠率
                |
            将分炼好的数据加载到追踪列表用MOSSE算法追踪
                |
            检查和更新目标
"""

import sys
import copy
import argparse
import cv2
import numpy as np
from keras.models import load_model
from utils.entity import Entity
from keras.applications.resnet50 import preprocess_input
import time



def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--file",
        help="Input video file.",
    )
    # Optional arguments.
    parser.add_argument(
        "--iou",
        default=0.00001,
        help="threshold for tracking",
    )
    args = parser.parse_args()
    track('video/one.flv', args.iou)


def overlap(box1, box2):
    """
    Check the overlap of two boxes
    检查两个框的重叠
    """
    endx = max(box1[0] + box1[2], box2[0] + box2[2])
    startx = min(box1[0], box2[0])
    width = box1[2] + box2[2] - (endx - startx)

    endy = max(box1[1] + box1[3], box2[1] + box2[3])
    starty = min(box1[1], box2[1])
    height = box1[3] + box2[3] - (endy - starty)

    if (width <= 0 or height <= 0):
        return 0
    else:
        Area = width * height
        Area1 = box1[2] * box1[3]
        Area2 = box2[2] * box2[3]
        ratio = Area / (Area1 + Area2 - Area)

        return ratio  # 重叠率


def track(video, iou):
    camera = cv2.VideoCapture(video)
    res, frame = camera.read()
    y_size = frame.shape[0]
    x_size = frame.shape[1]
    # Load CNN classification model加载CNN分类模型
    model = load_model('weights_body5.h5')
    """
    history：用于训练背景的帧数，默认为500帧，如果不手动设置learningRate，history就被用于计算当前的learningRate，此时history越大，learningRate越小，背景更新越慢；
    varThreshold：方差阈值，用于判断当前像素是前景还是背景。一般默认16，如果光照变化明显，如阳光下的水面，建议设为25, 36，具体去试一下也不是很麻烦，值越大，灵敏度越低；
    detectShadows：是否检测影子，设为true为检测，false为不检测，检测影子会增加程序时间复杂度，如无特殊要求，建议设为false
    """  # Definition of MOG2 Background SubtractionMOG2背景减法的定义
    bs = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    #cv2.imshow("bs", bs)
    history = 20
    frames = 0
    counter = 0
    check_time = 0
    # 设置帧率
    fps = 30
    # 获取窗口大小
    size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 调用VideoWrite（）函数
    videoWrite = cv2.VideoWriter('MySaveVideo.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    track_list = []
    # cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
    while True:
        res, frame = camera.read()

        if not res:
            break
        # Train the MOG2 with first frames frame用第一帧帧训练MOG2
        fg_mask = bs.apply(frame)

        if frames < history:
            frames += 1
            continue
        start = time.time()


        # frame原帧的扩展与去噪
        # 二值化阈值处理，前景掩码含有前景的白色值以及阴影的灰色值，在阈值化图像中，将非纯白色（244~255）的所有像素都设为0，而不是255
        th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow("th", th)
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        # 形态学膨胀
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        cv2.imshow("dilated", dilated)
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 确定回归边框
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            space = cv2.contourArea(c)
            if space > 1500 and h > w:  # 计算轮廓面积
                # Extract roi提取roi值
                img = frame[y: y + h, x: x + w, :]
                rimg = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)  # 图像缩放（64*64）
                image_data = np.array(rimg, dtype='float32')
                image_data /= 255.
                roi = np.expand_dims(image_data, axis=0)
                cv2.imshow("roi", image_data)
                flag = model.predict(roi)  # 输出预测结果

                if flag[0][0] > 0.9:
                    e = Entity(counter, (x, y, w, h), frame)

                    # 排除跟踪列表中的现有目标
                    if track_list:
                        count = 0
                        num = len(track_list)
                        for p in track_list:
                            if overlap((x, y, w, h), p.windows) < iou:
                                count += 1
                        if count == num:
                            track_list.append(e)
                    else:
                        track_list.append(e)
                    counter += 1

        # 检查和更新目标
        if track_list:
            tlist = copy.copy(track_list)
            lis_len = len(tlist)
            for enu in tlist:
                x, y = enu.center
                check_time += 1
                if 30 < x < x_size - 30 and 30 < y < y_size - 30:
                    enu.update(frame)
                    if lis_len > 5 and check_time < 2:
                        track_list.remove(enu)

                else:
                    track_list.remove(enu)



            end = time.time()
            seconds = (end - start) + 0.0001  # 保证seconds不为0
            fbs_cur = 1 / seconds
            fbs_show = "fbs: %.2f predict: %.2f goal:%d" % (fbs_cur, flag[0][0],check_time)
            cv2.putText(frame, fbs_show, (0, 50), cv2.FONT_HERSHEY_PLAIN,
                        1.5, (0, 0, 255), thickness=1)  # 图片，添加的文字，左下角坐标，字体，字体大小，颜色，字体粗细
            check_time = 0
            if seconds < 1.0 / fps:
                # 睡到30帧的频率，之后再往下进行
                time.sleep(1.0 / fps - seconds)
        cv2.imshow("detection", frame)
        # videoWrite.write(frame)
        key = cv2.waitKey(1) & 0xFF
        # 按' '键退出循环
        if key == ord(' '):
            break
    camera.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
