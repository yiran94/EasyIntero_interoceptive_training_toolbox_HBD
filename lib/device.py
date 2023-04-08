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
#TODO: fix ipcam
#import urllib2, base64
import numpy as np

class ipCamera(object):

    def __init__(self,url, user = None, password = None):
        self.url = url
        auth_encoded = base64.encodestring('%s:%s' % (user, password))[:-1]

        self.req = urllib2.Request(self.url)
        self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        response = urllib2.urlopen(self.req)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return frame

class Camera(object):

    def __init__(self, camera = 0):   #cam valid shape 这三个是Camera这个类的attribute
        self.cam = cv2.VideoCapture(camera)       # 括号里也可以加一个路径链接，就可以直接读取这个文件cv2.VideoCapture('test.mp4')，如果是数字的话就是用数字来控制不同的设备，例如0,1。
        self.valid = False
        #print('Frame rate of camera %d is %.2f fps ' % (camera,self.cam.get(cv2.CAP_PROP_FPS)) ) #打印视频帧率
        try:
            resp = self.cam.read()     #  cv2.VideoCapture()的对象可以用read这个methods，进行读取一帧图像。
            # resp得到的是一个元组，（True/False，一个三维数组），这个三维数组是第一帧的所有像素      
            self.shape = resp[1].shape #传递第一帧的尺寸，比如1080*1920*3
            #print('Resolution of this camera:'+str(self.shape))
            self.valid = True
        except:
            self.shape = None

    def get_frame(self):     # 获得当前帧
        if self.valid:
            _,frame = self.cam.read()
        else:
            frame = np.ones((480,640,3), dtype=np.uint8)
            col = (0,256,256)
            cv2.putText(frame, "(Error: Camera not accessible)",
                       (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame

    def release(self):
        self.cam.release()





