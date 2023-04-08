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
import cv2, time
import numpy as np
from scipy.signal import find_peaks

"""
Wraps up some interfaces to opencv user interface methods (displaying
image frames, event handling, etc).

If desired, an alternative UI could be built and imported into interoception_training.py 
instead. Opencv is used to perform much of the data analysis, but there is no
reason it has to be used to handle the UI as well. It just happens to be very
effective for our purposes.
"""

def imshow(*args,**kwargs):
    return cv2.imshow(*args,**kwargs)

def waitKey(*args,**kwargs):
    return cv2.waitKey(*args,**kwargs)

"""
The rest of this file defines some GUI plotting functionality. There are plenty
of other ways to do simple x-y data plots in python, but this application uses 
cv2.imshow to do real-time data plotting and handle user interaction.

This is entirely independent of the data calculation functions, so it can be 
replaced in the interoception_training.py application easily.
"""

def findbeat_build_in(data,background,count,flag,fps,buffer_size,i=0):
    z=background
    # print('buffer_size:%d' % buffer_size)
    w = min(int(z.shape[1]/3),int(z.shape[0]/2)) #shape[1]=1280,shape[0]=720

    cv2.rectangle(z,(z.shape[1]-w,i*w),(z.shape[1],(i+1)*w),(0,0,0),-1)  #在屏幕右上角画个黑底正方形,边长为w
    #i = 0 #把图画在画面上下的上部，这个参数可以根据需要画的位置调整，1是画在下部
    # graph_size=700  
    # z = np.zeros((graph_size,graph_size,3))
    peaks, _ = find_peaks(data,prominence=0.1)
    # print(peaks)
    for j in range(8):#在当前buffer size里的后8帧滑动寻找峰值点，取8是根据prominence来的，prominence会带来峰值判断的滞后，越高的p值滞后时间越长，0.1的p值大约滞后2-3帧，但是！！有时候
        #峰值点后面几帧由于噪声一直没有降低数值，以至于迟迟无法判断为峰值，有时候会延迟6帧才会判断出来，因此保险起见，留点buffer，取8。
        #取后8帧，实际造成的delay的时间GAP是8-1=7个GAP，一个GAP是33ms（1000ms除以30f），那么就会最多delay231ms
        if len(peaks)>0: #如果peaks是空集，那么下一行的peaks[-1]会报错
            if peaks[-1]==buffer_size-1-j: #如果此帧为峰值帧。这里59是buffer size=60再减1，因为最后一帧ID是59.
                # print(round(fps*60/120))
                if count-flag>round(fps*60/120): #限制必须这次心跳是和上次心跳间隔了n帧以上，在这个实验中我们认为人的极限心跳是120下每分钟，那么一个心跳周期就是占据了n(=fps 乘 60s 除以 120）帧
                    cv2.circle(z,(int(z.shape[1]-w/2),int((i+1/2)*w)),int(w/4),(0,0,255),-1) #展示心跳，半径w/4
                    # print('跳了：1')
                    flag=count #记下当前心跳是哪一帧
                    break #结束for 循环
    return count,flag


def plotXY_build_in(data,background,margin = 25,i=2):
    z= background
    w = float(z.shape[1]/2) #把画面分成左右两等份
    h = z.shape[0]/3  #把画面分成上中下三等份


    cv2.rectangle(z,(0,int(i*h)),(int(w),z.shape[0]),(0,0,0),-1)  #在屏幕左下角画个黑底矩形
    #i = 2 #把曲线图画在画面上中下的下部，这个参数可以根据需要画的位置调整，1是画在中部，0是顶部

    for x,y in data:
        # len(x)=data_butter的大小，len(y)=data_butter的大小
        if len(x) < 2 or len(y) < 2:
            col = (255, 255, 255)
            cv2.putText(z, "Loading...",(30, int(25+i*h)), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
            return #帧数不足时候直接返回，不画曲线图
    
    P = []
    for x,y in data:
        x = np.array(x) # x is time
        y = -np.array(y) # 【重要！！】计算机绘图的坐标系统和我们的纸笔作图经验相反，计算机的左上角是（0，0）原点，越向下y值越大。而我们的纸笔作图是左下角是原点，越向上y值越大，所以这里需要通过一个负号来抵消这里的日常作图习惯和计算机习惯的矛盾
        peaks, _ = find_peaks(-y,prominence=0.1)
        xx = (w-2*margin)*(x - x.min()) / (x.max() - x.min())+margin
        yy = (h-2*margin)*(y - y.min()) / (y.max() - y.min())+margin + i*h

        try:
            pts = np.array([[x_, y_] for x_, y_ in zip(xx,yy)],np.int32)
            P.append(pts)
        except ValueError:
            pass #temporary
    """ 
    #Polylines seems to have some trouble rendering multiple polys for some people
    for p in P:
        cv2.polylines(z, [p], False, (255,255,255),1)
    """
    #hack-y alternative:

    for p in P:
        for i in range(len(p)-1):
            #print(p[i][1])
            cv2.line(z,tuple(p[i]),tuple(p[i+1]), (255,255,255),1)
        # for i in range(len(p)-21): #find the max peak and label it with red dot
        #     if (p[i+10][1]-p[i][1])>0 and (p[i+10][1]-p[i+8][1])>0 and (p[i+10][1]-p[i+11][1])>0 and (p[i+10][1]-p[i+20][1])>0:
        #         cv2.circle(z,tuple(p[i+10]),1,(0,0,225),-1)
        for i in peaks:
            cv2.circle(z,tuple(p[i]),2,(225,225,225),-1)
    return z


