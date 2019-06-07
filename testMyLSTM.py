#import numpy as np
import os
import torch
import torch.nn.functional as F
#import torch.nn as nn
import time
import cv2
from myLSTM import getVideoFrames,myConvLSTM,getData

def saveVideoFrames(oldDir,newDir,bbs,newSize,frameLimit=None):
	#TODO: include ground-truth as args, compute IoU or some other loss
    frameFiles = os.listdir('./'+oldDir)
    frameFiles.sort()
    if (frameLimit is not None) or (frameLimit != -1):
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
    trainingList = ['Girl','Girl3','Girl2','Jumping','DragonBaby']
    testingList = ['BlurFace','Trellis','Dudek']#['BlurFace','Dudek','Dudek2','Girl2','Girl3','Jumping'] 
    trainingData = dict()
    testingData = dict()
    #newSize = 300 
    newSize= 180
    '''for data in trainingList:
        trainingData[data] = getVideoFrames(directory+data+'/img/',newSize)
    for data in testingList:
        testingData[data] = getVideoFrames(directory+data+'/img/',newSize)'''

    loadData = getData(directory,[],newSize)
    #Load model as lstm
    lstm = myConvLSTM((newSize,newSize),3,1,3) #changed for 1 or 3 channel
    #modelPath = './ConvLSTM_ADAMsl1loss/'
    modelPath = '/mnt/cloudNAS2/constantine/ConvLSTM_180size_s1loss_1channel/'
    #modelPath = '/mnt/cloudNAS2/constantine/ConvLSTM_180size_s1loss/'
    #modelPath = '/mnt/cloudNAS2/constantine/ConvLSTM_300size_s1loss/'
    #lstm.load_state_dict(torch.load(modelPath+'secondConvLSTM_250epochs_2019-06-01_06:41.pth'))
    #lstm.load_state_dict(torch.load(modelPath+'secondConvLSTM_250epochs_2019-06-05_08:29.pth')) #smooth l1 loss (best yet)
    #lstm.load_state_dict(torch.load(modelPath+'secondConvLSTM_250epochs_2019-06-05_20:22.pth')) #MSE loss
    lstm.load_state_dict(torch.load(modelPath+'secondConvLSTM_250epochs_180size_2019-06-06_07:33.pth')) #1 channel
    #lstm.load_state_dict(torch.load(modelPath+'secondConvLSTM_250epochs_300size_2019-06-06_20:05.pth')) # size 300
    lstm.eval()
    lstm.cuda()
    frameLimit= 360 #for systems with less GPU memory
    with torch.no_grad():
        avgLoss = 0
        for key in trainingList:
            print(key)
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
            print("Inference Time: ", tock-tick)
            seq_size = bbs.shape[0]
            with torch.enable_grad():
                sumLoss = 0
                for t in range(seq_size):
                    #bbs[t,:].requires_grad=True
                    nbb = loadData.getNextBox()
                    #nbb.requires_grad = True
                    loss =  F.smooth_l1_loss(torch.div(bbs[t,:],1.0),torch.div(nbb,1.0),reduction='mean')
                    #loss.require_grad=True
                    #loss.backward(retain_graph=True)
                    sumLoss += loss.data.item()
                    
            sampleLossAvg = sumLoss/seq_size
            print(sampleLossAvg)
            avgLoss += sampleLossAvg
            del loss
            del bbs

        print("Avg Training Loss : ", avgLoss/len(trainingList))
        avgLoss = 0
        for key in testingList:
            print(key)
            tick = time.time()
            #video_input = torch.unsqueeze(testingData[key][0],0).cuda()
            loadData.setVid(key,frameLimit)        
            #print("Before forward pass: ",torch.cuda.memory_allocated(torch.cuda.current_device()))
            bbs = lstm(loadData,loadData.getFirstBox())
            if not os.path.exists(directory+key+'/test/'):
                os.makedirs(directory+key+'/test/')
            saveVideoFrames(directory+key+'/img/',directory+key+'/test/',bbs,newSize,frameLimit)
            tock = time.time()
            print("Inference Time: ", tock-tick)
            seq_size = bbs.shape[0]
            with torch.enable_grad():
                sumLoss = 0
                for t in range(seq_size):
                    #bbs[t,:].requires_grad=True
                    nbb = loadData.getNextBox()
                    #nbb.requires_grad = True
                    loss =  F.smooth_l1_loss(torch.div(bbs[t,:],1.0),torch.div(nbb,1.0),reduction='mean')
                    #loss.requires_grad = True
                    #loss.backward(retain_graph=True)
                    sumLoss += loss.data.item()

            sampleLossAvg = sumLoss/seq_size
            print(sampleLossAvg)
            avgLoss += sampleLossAvg
            del loss
            del bbs

        print("Avg Testing Loss : ", avgLoss/len(testingList))
		    
			    
