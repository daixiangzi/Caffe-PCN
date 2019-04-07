import numpy as np
import cv2
import caffe
import math
import time

EPS = 1e-5
M_PI = 3.14159265358979323846
class PCN:
    def __init__(self):
        self.minface=""
        self.factor = ""
        self.scale =""
        self.stride = 8
        self.anglerange = 45
        self.augscale = 0.15
        self.thresh1 = 0.37
        self.thresh2 = 0.43
        self.thresh3 = 0.97
        self.nms_thresh1=0.8
        self.nms_thresh2 = 0.8
        self.nms_thresh3 = 0.3
        self.angle_range=45
        self.prelist = []
    def setmin_facesize(self,minfaces):
        minfaces = minfaces if minfaces>20 else 20
        minfaces *=1.4
        self.minface = minfaces
    def setimage_pyramidscalefactor(self,factors):
        self.scale = factors

    def legal(self,x, y, img):
        r, c, _ = img.shape
        if x >= 0 and x < c and y > 0 and y < r:
            return True
        else:
            return False
    def trans_windows(self,img,imgpad,winlist):#winlist:x,y,w,h,angle,scale,conf
        r = (imgpad.shape[0]-img.shape[0])/2
        c = (imgpad.shape[1]-img.shape[1])/2
        ret = []
        for i in range(len(winlist)):
            if winlist[i][2]>0 and winlist[i][3]>0:
                ret.append([winlist[i][0]-c,winlist[i][1]-r,winlist[i][3],winlist[i][4],winlist[i][6]])
        return ret
    def smooth_windows(self,winlist):
        for i in range(len(winlist)):
            for j in range(len(self.prelist)):
                if iou(winlist[i],self.prelist[j])>0.9:
                    winlist[i][5]=(winlist[i][5]+self.prelist[i][5])/2
                    winlist[i][0] = self.prelist[j][0]
                    winlist[i][1] = self.prelist[j][1]
                    winlist[i][2] = self.prelist[j][2]
                    winlist[i][3] = self.prelist[j][3]
                    winlist[i][4] = self.prelist[j][4]
                elif iou(winlist[i],self.prelist[j])>0.6:
                    winlist[i][5]=(winlist[i][5]+self.prelist[i][5])/2
                    winlist[i][0] = (self.pprelist[j][0]+winlist[j][0])/2
                    winlist[i][1] = (self.prelist[j][1]+winlist[j][1])/2
                    winlist[i][2] = (self.prelist[j][2]+winlist[j][2])/2
                    winlist[i][3] = (self.prelist[j][3]+winlist[j][3])/2
                    winlist[i][4] = smoothangle(winlist[j][4],self.prelist[j][4])
        prelist = winlist
        return winlist

    def stage1(self, img, imgpad, net, thres):
        r, c, _ = img.shape
        r_pad, c_pad, _ = imgpad.shape
        real_c = (c_pad - c) / 2
        real_r = (r_pad - r) / 2
        netsize = 24
        winlist = []
        curscale = self.minface/float(netsize)
        img_resize = resize_img(img,curscale)
        while min(img_resize.shape[0],img_resize.shape[1])>=netsize:
            prob,rotate_prob,reg = setinput(preprocess_img(img_resize),net,1)
            _,__,prob_h,prob_w= prob.shape
            w = netsize*curscale
            for i in range(prob_h):
                    for j in range(prob_w):
                        if prob[0,1,i,j]>thres:
                            sn = reg[0,0,i,j]
                            xn = reg[0,1,i,j]
                            yn = reg[0,2,i,j]
                            rx = int(j*curscale*self.stride-0.5*sn*w+sn*xn*w+0.5*w)+real_c
                            ry = int(i*curscale*self.stride-0.5*sn*w+sn*yn*w+0.5*w)+real_r
                            rw = int(w*sn)
                            if self.legal(rx,ry,imgpad) and self.legal(rx+rw-1,ry+rw-1,imgpad):
                                if rotate_prob[0,1,i,j]>0.5:
                                    winlist.append([rx,ry,rw,rw,0,curscale,prob[0,1,i,j]])
                                else:
                                    winlist.append([rx, ry, rw, rw, 180, curscale, prob[0, 1, i, j]])
            img_resize = resize_img(img_resize,self.scale)
            curscale = float(r)/(img_resize.shape[0])
        return winlist

    def stage2(self, img, img180, net, thres,dim,winlist):
        if len(winlist)==0:
            return winlist
        winlist = np.array(winlist)
        datalist=[]
        h = img.shape[0]
        for i in range(len(winlist)):
            if abs(winlist[i][4])<EPS:
                crop_img = img[int(winlist[i][1]):int(winlist[i][1]+winlist[i][3]),int(winlist[i][0]):int(winlist[i][0]+winlist[i][2])]
                datalist.append(preprocess_img(crop_img,dim))
            else:
                y2 = winlist[i][1] + winlist[i][3] - 1
                crop_img = img180[int(h-1-y2):int(winlist[i][3]+h-1-y2),int(winlist[i][0]):int(winlist[i][0]+winlist[i][2])]
                datalist.append(preprocess_img(crop_img, dim))
        datalist = np.array(datalist)
        prob, rotate_prob, reg = setinput(datalist, net,2)
        ret =[]
        for i in range(len(winlist)):
            if prob[i,1]>thres:
                sn = reg[i,0]
                xn =reg[i,1]
                yn = reg[i,2]
                cropx = winlist[i][0]
                cropy = winlist[i][1]
                cropw = winlist[i][2]
                if abs(winlist[i][4])>EPS:
                    cropy = h - 1-(cropy+cropw-1)
                w = int(sn*cropw)
                x = int(cropx-0.5*sn*cropw+cropw*sn*xn+0.5*cropw)
                y = int(cropy-0.5*sn*cropw+cropw*sn*yn+0.5*cropw)
                maxRotatescore = 0
                maxRotateindex = 0
                for j in range(3):
                    if rotate_prob[i,j]>maxRotatescore:
                        maxRotatescore = rotate_prob[i,j]
                        maxRotateindex = j
                if self.legal(x,y,img) and self.legal(x+w-1,y+w-1,img):
                    angle =0
                    if abs(winlist[i][4])<EPS:
                        if maxRotateindex ==0:
                            angle = 90
                        elif maxRotateindex == 1:
                            angle = 0
                        else:
                            angle = -90
                        ret.append([x,y,w,w,angle,winlist[i][5],prob[i,1]])
                else:
                    if maxRotateindex==0:
                        angle=90
                    elif maxRotateindex==1:
                        angle = 180
                    else:
                        angle = -90
                    ret.append([x,h-1-(y+w-1),w,w,angle,winlist[i][5],prob[i,1]])
        return ret

    def stage3(self,img,img180,img90,img_neg90,net,thres,dim, winlist):
        if len(winlist)==0:
            return winlist
        datalist=[]
        h,width,_ = img.shape
        for i in range(len(winlist)):#0:x,y:1,w:2,h:3
            if abs(winlist[i][4])<EPS:
                temp_img = img[winlist[i][1]:winlist[i][1]+winlist[i][3],winlist[i][0]:winlist[i][0]+winlist[i][2]]
                datalist.append(preprocess_img(temp_img,dim))
            elif abs(winlist[i][4]-90)<EPS:
                temp_img = img90[winlist[i][0]:winlist[i][0] + winlist[i][2], winlist[i][1]:winlist[i][1] + winlist[i][3]]
                datalist.append(preprocess_img(temp_img, dim))
            elif abs(winlist[i][4]+90) < EPS:
                x = winlist[i][1]
                y = width-1-(winlist[i][0]+winlist[i][2]-1)
                temp_img = img_neg90[y:y+winlist[i][3],x:x+winlist[i][2]]

                datalist.append(preprocess_img(temp_img, dim))
            else:
                y2 = winlist[i][1]+winlist[i][3]-1
                y = h -1-y2
                temp_img = img180[y:y + winlist[i][3], winlist[i][0]:winlist[i][0]+ winlist[i][2]]
                datalist.append(preprocess_img(temp_img, dim))
        datalist = np.array(datalist)
        prob, rotate_prob, reg = setinput(datalist, net,3)
        ret =[]
        for i in range(len(winlist)):
            if prob[i][1]>thres:
                img_tmp = img
                sn = reg[i][0]
                xn = reg[i][1]
                yn = reg[i][2]
                cropx = winlist[i][0]
                cropy = winlist[i][1]
                cropw = winlist[i][2]
                if abs(winlist[i][4]-180)<EPS:
                    cropy = h -1 -(cropy+cropw-1)
                    img_tmp = img180
                elif abs(winlist[i][4] - 90) < EPS:
                    cropy,cropx = cropx, cropy
                    img_tmp = img90
                elif abs(winlist[i][4] + 90) < EPS:
                    cropx = winlist[i][1]
                    cropy = width - 1 - (winlist[i][0] + winlist[i][2]-1)
                    img_tmp = img_neg90
                w = int(sn*cropw)
                x = int(cropx-0.5*sn*cropw+cropw*sn*xn+0.5*cropw)
                y = int(cropy-0.5*sn*cropw+cropw*sn*yn+0.5*cropw)
                angle = self.angle_range*rotate_prob[i][0]
                if legal(x,y,img_tmp) and legal(x+w-1,y+w-1,img_tmp):
                    if abs(winlist[i][4])<EPS:
                        ret.append([x,y,w,w,angle,winlist[i][5],prob[i][1]])
                    elif abs(winlist[i][4]-180)<EPS:
                        ret.append([x,h-1-(y+w-1),w,w,180-angle,winlist[i][5],prob[i][1]])
                    elif abs(winlist[i][4]-90)<EPS:
                        ret.append([y,x,w,w,90-angle,winlist[i][5],prob[i][1]])
                    else:
                        ret.append([width-y-w,x,w,w,angle-90,winlist[i][5],prob[i][1]])
        return ret

    def detect(self,img,imgpad,net):
        img180 = cv2.flip(imgpad,0)
        img90 = cv2.transpose(imgpad)
        img_neg90 = cv2.flip(img90,0)
        winlist = self.stage1(img,imgpad,net[0],self.thresh1)
        winlist = nms(winlist,True,self.nms_thresh1)

        winlist = self.stage2(imgpad,img180,net[1],self.thresh2,24,winlist)
        winlist = nms(winlist,True,self.nms_thresh2)

        winlist = self.stage3(imgpad,img180,img90,img_neg90,net[2],self.thresh3,48,winlist)
        winlist = nms(winlist,False,self.nms_thresh3)
        winlist = FP(winlist)
        return winlist
    def pcn_detect(self,img,net):
        imgpad = pad_img(img)
        winlist = self.detect(img,imgpad,net)
        return self.trans_windows(img,imgpad,winlist)


def pad_img(img):
        r,c,_ = img.shape
        r = min(int(r*0.2),100)
        c = min(int(c*0.2),100)
        ret = cv2.copyMakeBorder(img,r,r,c,c,cv2.BORDER_CONSTANT,value=(104, 117, 123))
        return ret

def preprocess_img(img,dim=0):
        if dim!=0:
            img_new = cv2.resize(img,(dim,dim))
        else:
            img_new = img
        mean = np.zeros(img_new.shape, np.uint8)+(104, 117, 123)
        return img_new -mean

def resize_img(img,scale):
        r,c,_ = img.shape
        ret = cv2.resize(img,(int(c/scale),int(r/scale)))
        return ret

def legal(x,y,img):
        r,c,_ = img.shape
        if x>=0 and x<c and y>0 and y<r:
            return True
        else:
            return False

def insert(x,y,rect):
        if x>rect[0] and y> rect[1] and x < (rect[0]+rect[2]) and y< (rect[1]+rect[3]):
            return True
        else:
            return False

def smoothangle(a,b):
        if a>b:
            a,b = b,a
        diff = (b-a)%360
        if diff <180:
            return a+diff/2
        else:
            return b+(360-diff)/2

def iou(w1,w2):
        x = max(0,min(w1[0]+w1[2]-1,w2[0]+w2[2]-1)-max(w1[0],w2[0])+1)
        y = max(0,min(w1[1]+w1[3]-1,w2[1]+w2[3]-1)-max(w1[1],w2[1])+1)
        inserts = x*y
        uino = w1[2]*w1[3] + w2[2]*w2[3] - inserts
        return float(inserts)/uino

def FP(winlist):
    if len(winlist)==0:
        return winlist
    ret = []
    winlist.sort(key=lambda x: x[6], reverse=True)
    flag = len(winlist) * [False]
    for i in range(len(winlist)):
        if flag[i]:
            continue
        for j in range(i+1,len(winlist)):
            if insert(winlist[i][0],winlist[i][1],winlist[i]) and insert(winlist[i][0]+winlist[i][2]-1,winlist[i][1]+winlist[i][3]-1,winlist[i]):
                flag[i] =True
        for i in range(len(winlist)):
            if not flag[i]:
                ret.append(winlist[i])
        return ret

def nms(winlist,local,threshold):
        if len(winlist)==0:
            return winlist
        rect2 = []
        winlist.sort(key=lambda x: x[6], reverse=True)
        flag = len(winlist)*[False]
        for i,_ in enumerate(flag):
                if _:
                    continue
                for j in range(i+1,len(winlist)):
                    if local and abs(winlist[i][5]-winlist[j][5])>1e-5:
                            continue
                    if iou(winlist[i],winlist[j])>threshold:
                        flag[j] = True
        for index,i in enumerate(flag):
                if not flag[index]:
                    rect2.append(winlist[index])
        return rect2

def setinput(img,net,flag):
    img = np.array(img)
    img = np.squeeze(img)
    if img.ndim==3:
        r, c, _ = img.shape
        b = 1
    if img.ndim == 4:
        b,r,c,_ = img.shape
    net.blobs['data'].reshape(b, 3,r,c)
    tmp_batch = np.zeros([b, 3, r,c], dtype=np.float32)
    if b==1:
        tmp_batch[0] = img.transpose(2, 0, 1).astype(np.float32)
    else:
        for i in range(b):
            tmp_batch[i]=img[i].transpose(2,0,1).astype(np.float32)
    net.blobs['data'].data[...]=tmp_batch
    net.forward()
    cls_prob =net.blobs['cls_prob'].data
    if flag==3:
        rotate_cls_prob = net.blobs['rotate_reg_3'].data
    else:
        rotate_cls_prob = net.blobs['rotate_cls_prob'].data
    flag = 'bbox_reg_%d'%flag
    bbox_reg_1 = net.blobs[flag].data
    return cls_prob,rotate_cls_prob,bbox_reg_1

def rotate_point(x,y,center_x,center_y,angle):
    x -=center_x
    y -=center_y
    theta = -angle*M_PI/180
    rx = int(center_x+x*math.cos(theta)-y*math.sin(theta))
    ry = int(center_y+x*math.sin(theta)+y*math.cos(theta))
    return (rx,ry)

def draw_line(img,point_list):
    thick =2
    cyan = (0,255,255)
    blue = (0,0,255)
    cv2.line(img,point_list[0],point_list[1],cyan,thick)
    cv2.line(img, point_list[1], point_list[2], cyan, thick)
    cv2.line(img, point_list[2], point_list[3], cyan, thick)
    cv2.line(img, point_list[3], point_list[0], blue, thick)
    return img

def draw_face(img,face):
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]+face[0]-1
    y2 = face[2]+face[1]-1
    centerx = (x1+x2)/2
    centery = (y1+y2)/2
    pointlist=[]
    pointlist.append(rotate_point(x1,y1,centerx,centery,face[3]))
    pointlist.append(rotate_point(x1, y2, centerx, centery, face[3]))
    pointlist.append(rotate_point(x2, y2, centerx, centery, face[3]))
    pointlist.append(rotate_point(x2, y1, centerx, centery, face[3]))
    draw_line(img,pointlist)

def crop_face(img, face, crop_size):
    x1,y1,x2,y2 = face[0],face[1],face[2]+face[0]-1,face[2]+face[1]-1
    center_x,center_y = (x1+x2)/2,(y1+y2)/2
    src,dst = [],[]
    src.append(rotate_point(x1,y1,center_x,center_y,face[3]))
    src.append(rotate_point(x1, y2, center_x, center_y, face[3]))
    src.append(rotate_point(x2, y2, center_x, center_y, face[3]))

    dst.append((0,0))
    dst.append((0, crop_size-1))
    dst.append((crop_size-1, crop_size-1))

    dst = np.float32(dst)
    src = np.float32(src)
    rot = cv2.getAffineTransform(src,dst)
    ret = cv2.warpAffine(img,rot,(crop_size,crop_size))
    return ret
def merge_imgs(imgA,imgB):
    if imgA == None:
        return imgB
    total_cols = imgA.shape[1]+imgB.shape[1]
    total_rows = max(imgA.shape[0],imgB.shape[0])
    ret = np.zeros((total_rows,total_cols,3),np.uint8)
    ret[:,:imgA.shape[1]]=imgA[:,:imgA.shape[1]]
    ret[:,imgA.shape[1]:] = imgB[:,:total_cols-imgA.shape[1]]
    return ret

if __name__ == "__main__":
    caffe.set_mode_cpu()  #set your computer mode,if you run on GPU maybe you should set caffe.set_model_gpu()
    caffemodel = './model/PCN.caffemodel' # set your caffe model files path
    net1_pro = './model/PCN-1.prototxt'
    net2_pro= './model/PCN-2.prototxt'
    net3_pro = './model/PCN-3.prototxt'
    net = []
    net.append(caffe.Net(net1_pro,caffemodel,caffe.TEST))
    net.append(caffe.Net(net2_pro,caffemodel,caffe.TEST))
    net.append(caffe.Net(net3_pro, caffemodel, caffe.TEST))
    pcn =PCN()
    pcn.setmin_facesize(20)
    pcn.setimage_pyramidscalefactor(1.414)
    cap = cv2.VideoCapture(0)
    while True:
        _,frame = cap.read()
        if not _:
            break
        begin_time = time.time()
        face = pcn.pcn_detect(frame, net)
        total_time = int(1/(time.time()-begin_time))
        str_time = str('fps:%d'%total_time)
        faceImg = None
        for i in range(len(face)):
            tmpFaceImg = crop_face(frame, face[i], 200)
            faceImg = merge_imgs(faceImg, tmpFaceImg)
            draw_face(frame,face[i])
        cv2.putText(frame, str_time, (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if len(face)!=0:
            cv2.imshow("crop", faceImg)
        cv2.imshow("raw", frame)
        if cv2.waitKey(33) == ord('q'):
            break
