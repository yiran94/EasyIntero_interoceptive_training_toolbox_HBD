'''
Copyright 2023 yiran94

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
from lib.device import Camera
from scipy import signal
from lib.frame_processor import FrameProcess
from lib.interface import imshow, waitKey
# from cv2 import moveWindow
import cv2
# import argparse
import numpy as np
import datetime
import sys
import time
import random
import os

class getPulseApp(object):

    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead.

    Then the average green-light intensity in the forehead region is gathered
    over time, and the detected person's pulse is estimated.
    """

    def __init__(self):
        # Imaging device - must be a connected camera (not an ip camera or mjpeg
        # stream)
        self.cameras = []
        self.selected_cam = 0
        for i in range(3):
            camera = Camera(camera=i)  # first camera by default   Camera这个类有三个attribute，详见device.py，那么camera也就有三个attribute，其中最主要的是shape，代表此摄像头的一帧图的尺寸
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera) #循环append，那么cameras这个list就会记录下你的系统所搭载的几个摄像头（最多是三个）的不同的像素尺寸（shape）配置
            else:
                break
        self.w, self.h = 0, 0
        self.pressed = 0
        
        # Containerized analysis of recieved image frames (an openMDAO assembly)
        # is defined next.

        # This assembly is designed to handle all image & signal analysis,
        # such as face detection, forehead isolation, time series collection,
        # heart-beat detection, etc.

        # Basically, everything that isn't communication
        # to the camera device or part of the GUI
        self.processor = FrameProcess(bpm_limits=[50, 160],
                                          data_spike_limit=2500.,
                                          face_detector_smoothness=10.)

        # Init parameters for the cardiac data plot
        self.reflection=False #Flag indicate whether under reflection model
        self.train=False #Flag indicate whether under train model

        # Maps keystrokes to specified methods
        #(A GUI window must have focus for these to work)
        self.key_controls = {"r": self.toggle_reflection,
                             "c": self.toggle_cam,
                             "t": self.toggle_train,
                             "y": self.answer_input,
                             "n": self.answer_input,
                             "1": self.confidence_input,
                             "2": self.confidence_input,
                             "3": self.confidence_input
                             }
        self.mapping = {"y": 0,   #作一个映射，y等同于没有delay也就是self.processor.train_answer这个参数的0
                        "n": 1}
        self.mapping2 = {"1": 1,
                         "2": 2,  
                         "3": 3}
        self.flag=0
        self.count=0
        self.t_start_reflection=0
        self.save_answer_judge=[]

    def toggle_cam(self):
        if self.processor.train_mode_state!=0: #在train模式下按C无效
            return
        if len(self.cameras) > 1: #首先判断现在有没有多个相机可切换
            self.reflection = False
            self.processor.reflection_mode = False
            self.processor.finish_reflection=False
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras) #取余数，防止self.selected_cam超过len(self.cameras)
            self.processor.switch_cam_frame=0 #记录一下切换相机的帧作为起点帧

    def toggle_train(self):
        if self.processor.finish_reflection:

            if self.processor.train_mode_state==0:
                            self.processor.train_mode_state=1 
                            return
            if self.processor.train_mode_state==1:
                            self.processor.train_mode_state=2
                            self.processor.train_answer=random.randint(0,1) #控制这个trial生成的是同步的心跳还是异步的心跳
                            self.processor.interoception_start_time=time.time()
                            self.processor.trial_id+=1
                            self.processor.save_tarining_data.append([]) #save_tarining_data最终应当是一个m维的list，m是train trail的次数，每进入一次train模式的一个trail，就增加一个list的子list，用于存储心跳数据
                            return
            if self.processor.train_mode_state==4:
                            self.processor.train_mode_state=2
                            self.processor.train_answer=random.randint(0,1)
                            self.processor.interoception_start_time=time.time() #记录开始的起点时间
                            self.processor.trial_id+=1
                            self.processor.save_tarining_data.append([])


    def answer_input(self):
        if self.processor.train_mode_state==3:
            self.save_answer_judge.append([self.processor.train_answer,self.mapping[chr(self.pressed)]]) #每次答完题存一下标准答案和作答答案
            if self.mapping[chr(self.pressed)]==self.processor.train_answer: #如果按下的按键对应的0/1映射和self.processor.train_answer一致
                self.processor.train_feedback=True #反馈这题答对了
                self.processor.train_correct_count+=1 #答对的题目数加1
            else:
                self.processor.train_feedback=False#反馈这题答错了


            self.processor.train_mode_state=3.5

    def confidence_input(self):
        if self.processor.train_mode_state==3.5:
            self.save_answer_judge[-1].append(self.mapping2[chr(self.pressed)]) #在标准答案和作答答案后面存一下confidence
            if self.processor.trial_id < self.processor.trial_num:
                    self.processor.train_mode_state=4
                    return
            else:   #存储数据及进入状态机5
                    self.save_answer_judge_arr=np.array(self.save_answer_judge)
                    self.processor.save_reflection_data_arr=np.array(self.processor.save_reflection_data)
                    self.processor.save_tarining_data_arr=np.array(self.processor.save_tarining_data)
                    path = 'Interoception_training_data/'
                    # 创建文件夹
                    if not os.path.exists(path):
                        os.makedirs(path)

                    fn = "Interoception_training_data_" + str(datetime.datetime.now())[0:19] #创建带时间戳的文件名
                    fn = fn.replace(":", "_").replace(".", "_").replace(" ", "_").replace("-", "_")
                    # np.save(fn,self.save_answer_judge_arr)
                    # np.savetxt(fn+"reflection.csv",self.processor.save_reflection_data_arr,delimiter=',')
                    # i=0
                    # for l in self.processor.save_tarining_data_arr:#这个三维array没办法存在一个csv文件里面，所以需要逐个存储
                        # i+=1
                        # np.savetxt(fn+"train_"+str(i)+".csv",l,delimiter=',')
                    np.savez(path+fn,reflection=self.processor.save_reflection_data_arr,judge = self.save_answer_judge_arr,train = self.processor.save_tarining_data_arr)
                    print('Train data saved!')
                    self.processor.train_mode_state=5 #进入状态机5

    def toggle_reflection(self):
        """
        Toggles the data display.
        """
        if self.processor.train_mode_state!=0: #在train模式下按R无效
            return
        if self.reflection:
            self.reflection = False
            self.processor.finish_reflection=False
            self.processor.reflection_mode = False
        else:
            if self.processor.light_condition: #       如果光线不达标按下R无效
                self.reflection = True
                self.processor.reflection_mode = True
                self.processor.finish_reflection=False
                self.processor.t_start_reflection=time.time()#记录进入reflection模式的起点时间，为1min倒计时做准备

    def key_handler(self):
        """
        Handle keystrokes, as set at the bottom of __init__()

        A plotting or camera frame window must have focus for keypresses to be
        detected.
        """
        self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            for cam in self.cameras:
                cam.cam.release()
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self):
        """
        Single iteration of the application's main loop. 
        一次循环，处理一帧
        """
        # Get current image frame from the camera
        frame = self.cameras[self.selected_cam].get_frame()  # 环宇：cameras is a list of several objects of class Camera, get_frame is a method of this class.
        self.h, self.w, _c = frame.shape

        # ！！！【核心解读】！！！
        # self是这个class定义的类：getPulseApp，self.processor 是另一个类：FrameProcess
        # self这个类主要包含了相机信息的获取和相机检测（通过调用Camera类实现），和main loop这个重要method
        # 后者self.processor这个类里包含了初始化（人脸识别的初始化等）和run（单帧的复杂处理）这个重要的method

        self.processor.frame_in = frame   #set current image frame to the processor's input，在main函数里有一个while True的循环不断调用这个loop函数，进而可以不断读取当前帧
        # process the image frame to perform all needed analysis
        self.processor.run(self.selected_cam)
        # collect the output frame for display
        output_frame = self.processor.frame_out # run函数处理后的帧


        # show the processed/annotated output frame
        imshow("Interoception Training", output_frame)
        self.key_handler()


if __name__ == "__main__":
    App = getPulseApp()  #构建class实例
    while True:
        App.main_loop()


