#import numpy as np
import os
import torch
#import torch.nn as nn
import time
import cv2
from myLSTM import getVideoFrames,myConvLSTM,getData

def saveVideoFrames(oldDir,newDir,bbs,newSize,frameLimit=None):
	#TODO: include ground-truth as args, compute IoU or some other loss
    frameFiles = os.listdir('./'+oldDir)
    frameFiles.sort()
    if frameLimit is not None:
        numFrames = frameLimit
    else:
        numFrames = len(frameFiles)
    #frames = np.zeros([numFrames,newSize,newSize,3])
    for i,frame in enumerate(frameFiles):
        img = cv2.imread('./'+oldDir+frame,cv2.IMREAD_COLOR)
        if i == 0: #probably cleaner way of doing this but not priority
            height, width, _ = img.shape
            heightFactor = height/newSize
            widthFactor = width/newSize
       
        bb = bbs[i,:]
		
        bb[0] = int(bb[0]*widthFactor)
        bb[1] = int(bb[1]*heightFactor)
        bb[2] = int(bb[2]*widthFactor)
        bb[3] = int(bb[3]*heightFactor)
        new = cv2.rectangle(img,(int(bb[0]),int(bb[1])),(int(bb[0])+int(bb[2]),int(bb[1])+int(bb[3])),(0,255,0),5)
		#save image in newDir
        cv2.imwrite('./'+newDir+frame, new)
        if i+1 == numFrames:
            break

	
if __name__=='__main__':
    directory = "../visual_tracker_benchmark/face/"
    trainingList = []#['Girl','Girl3','Girl2','Jumping','DragonBaby']
    testingList = ['BlurFace','Trellis','Dudek']#['BlurFace','Dudek','Dudek2','Girl2','Girl3','Jumping'] 
    trainingData = dict()
    testingData = dict()
    newSize = 100 
    '''for data in trainingList:
        trainingData[data] = getVideoFrames(directory+data+'/img/',newSize)
    for data in testingList:
        testingData[data] = getVideoFrames(directory+data+'/img/',newSize)'''

    loadData = getData(directory,[],newSize)
    #Load model as lstm
    lstm = myConvLSTM((newSize,newSize),3,3,3)
    #modelPath = './ConvLSTM_ADAMsl1loss/'
    modelPath = '../'
    #lstm.load_state_dict(torch.load(modelPath+'secondConvLSTM_250epochs_2019-06-01_06:41.pth'))
    lstm.load_state_dict(torch.load(modelPath+'secondConvLSTM_250epochs_2019-06-03_12:54.pth'))
    lstm.eval()
    lstm.cuda()
    frameLimit= 360 #for systems with less GPU memory
    for key in trainingList:
        tick = time.time()
        '''if key=='Dudek':
            video_input = torch.unsqueeze(trainingData[key][0],0)
        else:
            video_input = torch.unsqueeze(trainingData[key][0],0).cuda()'''
        loadData.setVid(key)
        #bbs = lstm(video_input,trainingData[key][1][0,:])
        bbs = lstm(loadData,loadData.getFirstBox())
        if not os.path.exists(directory+key+'/test/'):
            os.makedirs(directory+key+'/test/')
        saveVideoFrames(directory+key+'/img/',directory+key+'/test/',bbs,newSize)
        tock = time.time()
        del bbs
        print(tock-tick)
	
    for key in testingList:
        tick = time.time()
        #video_input = torch.unsqueeze(testingData[key][0],0).cuda()
        loadData.setVid(key,frameLimit)        
        #print("Before forward pass: ",torch.cuda.memory_allocated(torch.cuda.current_device()))
        bbs = lstm(loadData,loadData.getFirstBox())
        if not os.path.exists(directory+key+'/test/'):
            os.makedirs(directory+key+'/test/')
        saveVideoFrames(directory+key+'/img/',directory+key+'/test/',bbs,newSize,frameLimit)
        tock = time.time()
        del bbs
        print(tock-tick)
		
			
