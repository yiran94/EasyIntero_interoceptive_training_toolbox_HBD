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
import numpy as np
import time
import cv2
import os
import sys
from lib.interface import plotXY_build_in,findbeat_build_in
import random
import datetime

class FrameProcess(object):

    def __init__(self, bpm_limits=[], data_spike_limit=250,
                 face_detector_smoothness=10):

        self.frame_in = np.zeros((10, 10))  # import numpy as np
        self.frame_out = np.zeros((10, 10)) # 构建10*10的array，数值都是0
        self.fps = 30 #初始化一个大致数字或者0

        self.buffer_size = 60 #不要轻易改这个数字，interface.py文件里面132行hardcode了这个60还没解决掉。
        #self.window = np.hamming(self.buffer_size)
        self.data_buffer = []
        self.data_buffer_correct = []
        self.data_buffer_delayed = []
        self.data_buffer_whole = [[0],[0],[0]]
        self.data_buffer_whole_delay = [[0],[0],[0]]
        self.times = []
        self.times_new = []
        self.ttimes = []
        self.samples = []
        self.samples_whole = [0,0,0]
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False

        self.idx = 1
        self.reflection_mode=False
        self.train_mode=False
        self.t_start_reflection=0
        self.t_current_reflection=0
        self.remaining_time=0
        self.finish_reflection=False
        self.train_answer=999
        self.train_mode_interocrption=False
        self.train_mode_wait_answer=False
        self.train_mode_rest=False
        self.train_mode_state=0 
        # 0:Not in train mode 
        # 1:introduction
        # 2:interoception ongoing 
        # 3:waiting answer input 
        # 4:rest and waiting command to start next trial
        # 5:finished 
        self.reflection_time=60 #在reflection模式需要reflect多少秒
        self.trial_num = 20 #在training阶段一共有多少次trail
        self.trial_duration = 10 #每个trial持续多少秒
        self.trial_id = 0 #记录当前在哪一次trial
        self.interoception_start_time=0
        self.count=0
        self.flag=0
        self.train_feedback=False
        self.save_reflection_data=[] #要被存储为文件保存的reflection模式下的心跳数据
        self.train_correct_count=0
        self.save_tarining_data=[] #要被存储为文件保存的train模式下的心跳数据
        self.switch_cam_frame=0 #记录一下切换相机的帧作为起点帧,初始化也算切换相机，这个参数用在本文件的run函数定义的开头几行
        self.light_condition=False

    def get_wholeframe_means(self):  #用手指图的整个frame的亮度均值来求pulse
        v1 = np.mean(self.frame_in[:, :, 0])
        v2 = np.mean(self.frame_in[:, :, 1])
        v3 = np.mean(self.frame_in[:, :, 2]) #red
        return v3

    def run(self, cam):
        self.times.append(time.time() - self.t0)  
        self.switch_cam_frame+=1
        if self.switch_cam_frame==30:    #切换摄像头过几帧之后（30是随意取的，不一定要是30），等摄像头稳定之后，计算一下最近10帧的平均FPS，作为新摄像头的FPS
            self.fps=10/(self.times[-1]-self.times[-11])

            print("FPS of current camera: %2f " % self.fps)

        #     print(self.times[-1]-self.times[-2])
        # self.t0记录的是摄像头初始化的时候的时间，在
        # 之后每读一帧，就通过time.time()记录下读取当下帧的时间
        # 我check了一下，两次读帧相差了0.033，基本上和帧率30fps是一致的，
        # 说明程序处理帧并没有delay读帧的速度，还是可以赶上接收的帧率
        UPPER=np.mean(self.frame_in[:,:,2])+4
        LOWER=np.mean(self.frame_in[:,:,2])-4

        vals_whole= self.get_wholeframe_means()
        self.data_buffer.append(vals_whole)

        L = len(self.data_buffer)
        delay=round(266/(1000/self.fps)) #控制异步信号的delay时间，266ms,计算一下相当于delay多少帧.虽然理论上这么多帧的总间隔应该是233ms，但实际观察中由于帧率不稳定，会发现有的时候这几帧间隔是210ms，有的是250ms，这个几十毫秒的误差还行，在我们的实验里可以接受。
        # if not self.reflection_mode:
            # print(delay)
        if L > self.buffer_size+delay:

            self.data_buffer = self.data_buffer[-self.buffer_size-delay:] #取databuffer里最新的n+delay帧的RGB均值数据
            self.times = self.times[-self.buffer_size:] #取最新的n帧的时间
            # print("%.3f" %(self.times[-1]-self.times[-8])) 

            self.data_buffer_correct = self.data_buffer[-self.buffer_size:]
            self.data_buffer_delayed = self.data_buffer[-self.buffer_size-delay:-delay]

        self.samples_correct = np.array(self.data_buffer_correct) # array的计算速度比list快。但是有一点不同，list.append 比np.append(arr, np.array(i))快。
        self.samples_delayed = np.array(self.data_buffer_delayed) # self.samples这两个量用来被其他函数调用绘制亮度变化曲线图

        self.frame_out=self.frame_in
        
        # self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
        #                                           cv2.COLOR_BGR2GRAY))#COLOR_BGR2GRAY是一种转换成灰色的方式/code
        #cv2.equalizeHist是做了直方图均衡，直方图均衡可以增加图像的明暗对比度，refer to: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
        #输出从720*1080*3变成了720*1080
        col = (255, 255, 255)
        col_red = (0,0,255)
        col_green = (0,255,0)
        w=self.frame_out.shape[1]
        h=self.frame_out.shape[0]
        # print(w,h)
        
        if self.train_mode_state==1:  # 1: introduction
            cv2.rectangle(self.frame_out,(0,0),(w,h),(0,0,0),-1)
            cv2.putText(self.frame_out, "Training Introduction:",
                        (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            cv2.putText(self.frame_out, "1.In this mode, you will take 20 trials.",
                        (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            cv2.putText(self.frame_out, "2.For each trial, you will see a simulated heart beating around 10 times",
                        (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            cv2.putText(self.frame_out, "  (Randomly synchronized or not with your heartbeat).",
                        (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            cv2.putText(self.frame_out, "3.Compare this simulated heatbeat with the perception of your",
                        (10, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            cv2.putText(self.frame_out, "  heartbeat. Judge Synchronized or not.",
                        (10, 180), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            cv2.putText(self.frame_out, "4.Before start, Please put your finger stick to the camera and",
                        (10, 210), cv2.FONT_HERSHEY_PLAIN, 1.5, col_green)
            cv2.putText(self.frame_out, "  ensure the oscillogram is regular.",
                        (10, 240), cv2.FONT_HERSHEY_PLAIN, 1.5, col_green)
            cv2.putText(self.frame_out, "Press T to start...",
                        (10, 270), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            cv2.putText(self.frame_out, "(Cautions: Anytime press 'Esc' will quit this program)",
                        (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, col)
            plotXY_build_in([[self.times,self.samples_correct]],background=self.frame_out)
            return

        if self.train_mode_state==2:  # 2:interoception ongoing 
            if time.time()-self.interoception_start_time<self.trial_duration: #控制每次trial内感知的时间
                cv2.rectangle(self.frame_out,(0,0),(w,h),(0,0,0),-1)
                cv2.putText(self.frame_out, "Training Mode:",
                            (400, 40), cv2.FONT_HERSHEY_PLAIN, 2, col)
                cv2.putText(self.frame_out, "Trial %d / %d" % (self.trial_id,self.trial_num),
                            (400, 80), cv2.FONT_HERSHEY_PLAIN, 2, col)
                cv2.putText(self.frame_out, "Hold your finger still",
                            (400, 120), cv2.FONT_HERSHEY_PLAIN, 2, col_green)
                if self.train_answer==0:
                    self.train_mode_display=self.samples_correct
                else:
                    self.train_mode_display=self.samples_delayed
                plotXY_build_in([[self.times,self.train_mode_display]],background=self.frame_out,i=1)
                self.count+=1
                self.count,self.flag=findbeat_build_in(data=self.train_mode_display,background=self.frame_out,count=self.count,flag=self.flag,fps=self.fps,buffer_size=self.buffer_size,i=0)
                self.if_shown_as_beat=int(self.count==self.flag)#如果self.count=self.flag，那说明在这一帧产生了心跳动画，int(True)=1
                # print(self.if_shown_as_beat)
                self.save_tarining_data[self.trial_id-1].append([time.time(),vals_whole,self.train_mode_display[-1],self.if_shown_as_beat])
            else: #  内感知时间到，进入待答题模式
                cv2.rectangle(self.frame_out,(0,0),(w,h),(0,0,0),-1)
                self.train_mode_state=3
            return

        if self.train_mode_state==3:        # 3:waiting answer input 
            cv2.rectangle(self.frame_out,(0,0),(w,h),(0,0,0),-1)
            cv2.putText(self.frame_out, "Trial %d / %d" % (self.trial_id,self.trial_num),
                            (100, 80), cv2.FONT_HERSHEY_PLAIN, 2, col)
            cv2.putText(self.frame_out, "Synchronize with your heartbeat?",
                        (100, int(h/4)), cv2.FONT_HERSHEY_PLAIN, 2, col)
            cv2.putText(self.frame_out, "Press Y or N",
                        (100, int(h/2)), cv2.FONT_HERSHEY_PLAIN, 2, col)
            return

        if self.train_mode_state==3.5:        # 3.5: waiting confidence input
            cv2.rectangle(self.frame_out,(0,0),(w,h),(0,0,0),-1)
            cv2.putText(self.frame_out, "Trial %d / %d" % (self.trial_id,self.trial_num),
                            (100, 80), cv2.FONT_HERSHEY_PLAIN, 2, col)
            cv2.putText(self.frame_out, "How much confidence do you have in the choice you just made?",
                        (100, int(2*h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)
            cv2.putText(self.frame_out, "Press '1':  Not at all/Total guess(No heartbeat awareness)",
                        (100, int(4*h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)
            cv2.putText(self.frame_out, "Press '2':  Moderate confidence",
                        (100, int(5*h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)
            cv2.putText(self.frame_out, "Press '3':  Complete confidence(Full perception of heartbeat/time)",
                        (100, int(6*h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)
            return

        if self.train_mode_state==4:        # 4:rest and waiting command to start next trial
            cv2.rectangle(self.frame_out,(0,0),(w,h),(0,0,0),-1)
            if self.train_feedback:
                if self.train_answer==0:
                    cv2.putText(self.frame_out, "Answer saved, your answer is Right! It's Synchronized",
                                (10, int(h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)
                else:
                    cv2.putText(self.frame_out, "Answer saved, your answer is Right! It's Not synchronized",
                                (10, int(h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)
            else:
                if self.train_answer==0:
                    cv2.putText(self.frame_out, "Answer saved, your answer is Wrong! It's Synchronized",
                                (10, int(h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)
                else:
                    cv2.putText(self.frame_out, "Answer saved, your answer is Wrong! It's Not synchronized",
                                (10, int(h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)

            cv2.putText(self.frame_out, "Before start, Please put your finger stick to the camera and",
                        (10, int(2*h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col_green)
            cv2.putText(self.frame_out, "ensure the oscillogram is regular.",
                        (10, int(2*h/8)+30), cv2.FONT_HERSHEY_PLAIN, 2, col_green)
            cv2.putText(self.frame_out, "Press T to continue",
                        (10, int(3*h/8+30)), cv2.FONT_HERSHEY_PLAIN, 2, col)
            plotXY_build_in([[self.times,self.samples_correct]],background=self.frame_out,i=2)
            return

        if self.train_mode_state==5:        # 5:finished 
            cv2.rectangle(self.frame_out,(0,0),(w,h),(0,0,0),-1)
            if self.train_feedback:
                if self.train_answer==0:
                    cv2.putText(self.frame_out, "Answer saved, your answer is Right! It's Synchronized",
                                (10, int(h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)
                else:
                    cv2.putText(self.frame_out, "Answer saved, your answer is Right! It's Not synchronized",
                                (10, int(h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)
            else:
                if self.train_answer==0:
                    cv2.putText(self.frame_out, "Answer saved, your answer is Wrong! It's Synchronized",
                                (10, int(h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)
                else:
                    cv2.putText(self.frame_out, "Answer saved, your answer is Wrong! It's Not synchronized",
                                (10, int(h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)
            self.train_accuracy=int((self.train_correct_count/self.trial_num)*100)
            cv2.putText(self.frame_out, "You have finished today's training. "+"("+str(datetime.datetime.now())[0:16]+")",
                        (10, int(2*h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col_green)
            cv2.putText(self.frame_out, "Training data saved.",
                        (10, int(3*h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col_green)
            cv2.putText(self.frame_out, "Today's Accuracy: %d %%" % self.train_accuracy,
                        (10, int(4*h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col_green)
            cv2.putText(self.frame_out, "Press 'ESC' to quit",
                        (10, int(5*h/8)), cv2.FONT_HERSHEY_PLAIN, 2, col)
            return

        if self.reflection_mode:
            cv2.rectangle(self.frame_out,(0,0),(w,h),(0,0,0),-1)

            plotXY_build_in([[self.times,self.samples_correct]],background=self.frame_out)
            self.count+=1
            self.count,self.flag=findbeat_build_in(data=self.samples_correct,background=self.frame_out,count=self.count,flag=self.flag,fps=self.fps,buffer_size=self.buffer_size)

            cv2.putText(self.frame_out, "Now in Reflection Mode(Takes 1min)",
                        (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            cv2.putText(self.frame_out, "1.Please try to percept your heartbeat without",
                        (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, col_green)
            cv2.putText(self.frame_out, "  Putting your hand on your chest",
                        (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, col_green)
            cv2.putText(self.frame_out, "2.Comparing your feeling with your being detected heartbeat",
                        (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            cv2.putText(self.frame_out, "  displayed on the right side for 1min",
                        (10, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            self.t_current_reflection=time.time()
            self.remaining_time=self.reflection_time-int(self.t_current_reflection-self.t_start_reflection)
            if self.remaining_time>0:
                cv2.putText(self.frame_out, "Remaining reflection time: %ds " % self.remaining_time,
                        (10, 210), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
                self.if_shown_as_beat=int(self.count==self.flag)#如果self.count=self.flag，那说明在这一帧产生了心跳动画，int(True)=1
                # print(self.if_shown_as_beat)
                self.save_reflection_data.append([time.time(),vals_whole,self.if_shown_as_beat])
            else:
                cv2.putText(self.frame_out, "Finished! Press 'T' to enter Training mode",
                        (10, 240), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
                self.finish_reflection=True

            cv2.putText(self.frame_out, "(Press 'R' to quit Reflection Mode, return to Welcome Page)",
                        (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, col)
            cv2.putText(self.frame_out, "(Any time Press 'Esc' will quit this program)",
                   (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, col)
            return


        #以下是程序运行后的起始模式展示的文字：
        cv2.rectangle(self.frame_out,(0,0),(w,h),(0,0,0),-1)
        cv2.putText(self.frame_out, "Now on Welcome Page",
                    (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "0.(Optional)Press 'C' to change camera (current: %s)" % str(cam),
                    (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "1.Put your finger on the camera with slight pressure",
                   (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, col)  
        cv2.putText(self.frame_out, "2.Adjust light condition in your room to",
                   (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, col_green)
        cv2.putText(self.frame_out, "  make sure your light condition is above 15",
                   (10, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, col_green)  
        cv2.putText(self.frame_out, " Current Light Condition: %d" % vals_whole,
                   (30, 180), cv2.FONT_HERSHEY_PLAIN, 1, col)
        
        if L > self.buffer_size+delay: 
            gap=self.data_buffer_correct[np.argmax(self.data_buffer_correct)] - self.data_buffer_correct[np.argmin(self.data_buffer_correct)]
            cv2.putText(self.frame_out, " Light Max-Min: %2f" % gap,
                   (30, 210), cv2.FONT_HERSHEY_PLAIN, 1, col)

        cv2.putText(self.frame_out, "3.Adjust your finger and light condition",
                   (10, 240), cv2.FONT_HERSHEY_PLAIN, 1.5, col_green)
        cv2.putText(self.frame_out, "  until seeing regular oscillogram",
                   (10, 270), cv2.FONT_HERSHEY_PLAIN, 1.5, col_green)         
        cv2.putText(self.frame_out, "4.Press 'R' to enter Reflection Mode",
                   (10, 300), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "(Any time Press 'Esc' will quit this program)",
                   (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, col)
        plotXY_build_in([[self.times,self.samples_correct]],background=self.frame_out)
        if vals_whole<10:
            self.light_condition=False
            cv2.putText(self.frame_out, "Warning:",
                   (int(w/4), int(h/2)), cv2.FONT_HERSHEY_PLAIN, 2, col_red)
            cv2.putText(self.frame_out, "Light condition around you is loewer than 10",
                   (int(w/4), int(h/2+30)), cv2.FONT_HERSHEY_PLAIN, 2, col_red)
            cv2.putText(self.frame_out, "Please increase illumination and then continue",
                   (int(w/4), int(h/2)+60), cv2.FONT_HERSHEY_PLAIN, 2, col_red)
        else:
            self.light_condition=True



