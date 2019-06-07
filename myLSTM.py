
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.variable import Variable




class myLSTM(nn.Module):
    #input_dim: ht+input
    #hidden_dim: just ht
    def __init__(self,input_dim, hidden_dim):
        
        self.ht_prev #TODO: initialize, size???
        #Previous cell state
        self.ct_prev #TODO: initialize, size???
        self.hidden_dim = hidden_dim
        #LSTM Layers:
        self.ft_out = nn.Sequential(nn.Linear(input_dim+hidden_dim,hidden_dim), nn.Sigmoid()) 
        self.it_out = nn.Sequential(nn.Linear(input_dim+hidden_dim,hidden_dim),nn.Sigmoid()) 
        self.ct_new_out = nn.Sequential(nn.Linear(input_dim+hidden_dim, hidden_dim), nn.Tanh())  #Why is tanh used here??? ("push values between -1 and 1" ???), since sigmoid is between 0 and 1, effectively normalizes???
        self.output = nn.Sequential(nn.Linear(input_dim+hidden_dim,hidden_dim),nn.Sigmoid()) 

    def forward(self,x):
        batch_size, seq_size, _ , _ , _= x.size()
        hidden_out = torch.zeros(self.hidden_dim,seq_size)
        ht = torch.zeros(self.hidden_dim,1)
        ct = torch.zeros(self.hidden_dim,1)
        for t in range(seq_size):
            x_t = x[:,t,:,:,:]
            #Concatenate input and previous output together
            combined = torch.cat((ht, x),0)
        
            #Cell state
            #forget gate : Control how much we remember by multiplying ct between 0 and 1
            ft = self.ft_out(combined)
        
            #input gate: Decide what will be updated
            it = self.it_out(combined) 
            # New cell state candidates
            ct_new = self.ct_new_out(combined) #Decide the actual amounts added to cell state
        
            #Update cell state with new candidate values and apply forget gate
            ct = (ct * ft) + (ct_new*it)
            
            #Output
            ot = self.output(combined)
            ht = ot * nn.Tanh(ct)

            #Save ht for this time step
            hidden_out[:,t] = ht
        '''#Save our newly calculated ct and ht as they are now the "previous" value for the next iteration
        self.ct_prev = ct
        self.ht_prev = ht
        #Return the output'''
        return hidden_out,(ht,ct)

class myConvLSTM(nn.Module):

    def __init__(self,shape,input_chan,hidden_chan,filter_size):
        super(myConvLSTM, self).__init__()
        self.shape = shape #height and width
        self.input_chan = input_chan
        self.hidden_chan = hidden_chan
        self.filter_size = filter_size
        self.padding = int((filter_size-1)/2) #allows output to have same size
        self.outputOps = nn.ModuleList([nn.Sigmoid(),nn.Sigmoid(),nn.Tanh(),nn.Sigmoid(), nn.Tanh()])
        self.conv = nn.Conv2d(self.input_chan + self.hidden_chan, 4*self.hidden_chan, self.filter_size,1,self.padding) #Effectively only need one Conv Layer, because output channels used for each of ft, it, and ct
        print(self.conv)
        #TODO: more processing to go from feature map to 4 coord bb???
        self.maxpool = nn.Sequential(nn.Conv2d(self.hidden_chan,1,self.filter_size,1,self.padding),nn.MaxPool2d(4,stride=4))
        self.bb_out = nn.Sequential(nn.Linear(int(self.shape[0]/4)*int(self.shape[1]/4),100),nn.Linear(100,4))
 
    def forward(self,loader,init_bb):
        batch_size = loader.batchSize
        seq_size = loader.numFrames
        new = cv2.rectangle(np.zeros((self.shape[0],self.shape[1],self.hidden_chan)),(int(init_bb[0]),int(init_bb[1])),(int(init_bb[0])+int(init_bb[2]),int(init_bb[1])+int(init_bb[3])),(0,255,0),5)
        new = torch.from_numpy(np.moveaxis(new,-1,0)).float()
        #hidden_out = torch.zeros(seq_size,self.hidden_chan, self.shape[0],self.shape[1]) #UNCOMMENT TO LOOK AT hidden feature map at each step
        hidden_bb = torch.zeros(seq_size,4)
        #print("Before cudaing ht and ct: ", torch.cuda.memory_allocated(torch.cuda.current_device()))
        ht = torch.unsqueeze(new,0)#torch.zeros(batch_size,self.hidden_chan, self.shape[0],self.shape[1])
        ht = ht.cuda() 
        ct = torch.zeros(batch_size,self.hidden_chan, self.shape[0],self.shape[1])
        ct = ct.cuda()
        for t in range(seq_size):
#            xt = x[:,t,:,:,:]
            xt = loader.getNextFrame()
            #print(t)
            #print("Before cudaing xt: ", torch.cuda.memory_allocated(torch.cuda.current_device()))
            xt = xt.cuda()
            #print("After cudaing xt: ",torch.cuda.memory_allocated(torch.cuda.current_device()))


            combined = torch.cat((xt, ht),1) #concatenate along channel dim
            conv_out = self.conv(combined)
            (ft_in, it_in, ct_new_in, ot_in) = torch.split(conv_out, self.hidden_chan, dim=1)
            ft = self.outputOps[0](ft_in)
            it = self.outputOps[1](it_in)
            ct_new = self.outputOps[2](ct_new_in)
            ot = self.outputOps[3](ot_in)

            ct = ft*ct + it*ct_new
            ht = ot * self.outputOps[4](ct)
            #Save ht for this time step
            #hidden_out[t,:,:,:] = ht #UNCOMMENT TO LOOK AT hidden feature map at each step
            ht_pool = self.maxpool(ht)
            bb = self.bb_out(ht_pool.reshape(ht_pool.size(0),-1))
            hidden_bb[t,:] = bb
            #print("After fwd pass: ",torch.cuda.memory_allocated(torch.cuda.current_device()))
            del xt
            del ot
            del ft
            del it
            del ct_new
            del ht_pool
            torch.cuda.empty_cache()
            #print("After freeing cache (I think)",torch.cuda.memory_allocated(torch.cuda.current_device()))
            #xt = 0
        del  ht
        del ct
        torch.cuda.empty_cache()
        return hidden_bb #put hidden_out as first argument if want to look at

            


def getVideoFrames(targetDir,newSize):
    frameFiles = os.listdir('./'+targetDir)
    frameFiles.sort()
    numFrames = len(frameFiles)
    frames = np.zeros([numFrames,3, newSize,newSize])
    bbs = np.zeros([numFrames,4])
    gt_fp = open('./'+targetDir+'/../groundtruth_rect.txt','r')
    for i,frame in enumerate(frameFiles):
        img = cv2.imread('./'+targetDir+frame,cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        img = cv2.resize(img,(newSize,newSize),interpolation=cv2.INTER_AREA)
        frames[i,:,:,:] = np.moveaxis(img,-1,0)
        bb_str = gt_fp.readline()
        bb_str = bb_str.split()
        if ',' in bb_str[0]: #inconsistent labeling: some use whitespace, others commas
            bb_str = bb_str[0].split(',')
        bb = [int(n) for n in bb_str]
        bbs[i,:] = np.array([int((bb[0]/width)*newSize),int((bb[1]/height)*newSize),int((bb[2]/width)*newSize),int((bb[3]/height)*newSize)])

    gt_fp.close()
    return torch.from_numpy(frames).float(),torch.from_numpy(bbs).float()

#FOR conv2d needs to be (Batch, Number Channels, height, width)
#Crude data loader for image sequences:
class getData():
    def __init__(self,srcDir,imgDirs,newSize):
        self.srcDir = srcDir
        self.imgDirs = imgDirs #List of possible videos, not sure if necessary???
        self.newSize = newSize
        self.currVid = ''
        self.frameFiles = []
        self.gt_fp = -1
        self.idx   = -1
        self.numFrames = -1
        self.width = -1
        self.height = -1
        self.batchSize = 1

    
    def setVid(self,vidName,frameLimit=None):
        if self.gt_fp != -1:
            self.gt_fp.close()
        self.idx = 0
        self.currVid = vidName
        self.frameFiles = os.listdir('./'+self.srcDir+self.currVid+'/img')
        self.frameFiles.sort()
        if frameLimit is not None:
            self.numFrames = frameLimit
        else:
            self.numFrames = len(self.frameFiles)
        self.gt_fp = open('./'+self.srcDir+self.currVid+'/groundtruth_rect.txt','r')
        img = cv2.imread('./'+self.srcDir+self.currVid+'/img/'+self.frameFiles[self.idx])
        self.height, self.width, _ = img.shape #Not sure if there's a simpler way of finding dimensions other than loading image and getting shape

    def getNextFrame(self):
        #Get next frame in vid based on self.idx (change self.idx if want to get a different frame
        currFrameStr = './'+self.srcDir+self.currVid+'/img/'+self.frameFiles[self.idx]
        self.idx += 1
        #print(currFrameStr)
        img = cv2.imread(currFrameStr,cv2.IMREAD_COLOR)
        img = cv2.resize(img,(self.newSize,self.newSize),interpolation=cv2.INTER_AREA)
        img = np.moveaxis(img,-1,0)
        
        img_torch =  torch.from_numpy(img).float()


        return torch.unsqueeze(img_torch,0)

    def getFirstBox(self):
        curr = self.gt_fp.tell()
        self.gt_fp.seek(0,0)
        bb_str = self.gt_fp.readline()
        bb_str = bb_str.split()
        if ',' in bb_str[0]: #inconsistent labeling: some use whitespace, others commas
            bb_str = bb_str[0].split(',')
        bb = [int(n) for n in bb_str]
        bb_np = np.array([int((bb[0]/self.width)*self.newSize),int((bb[1]/self.height)*self.newSize),int((bb[2]/self.width)*self.newSize),int((bb[3]/self.height)*self.newSize)])
        bb_torch = torch.from_numpy(bb_np).float()

        self.gt_fp.seek(curr,0)
        return bb_torch
    
    def getNextBox(self):
        bb_str = self.gt_fp.readline()
        bb_str = bb_str.split()
        if ',' in bb_str[0]: #inconsistent labeling: some use whitespace, others commas
            bb_str = bb_str[0].split(',')
        bb = [int(n) for n in bb_str]
        bb_np = np.array([int((bb[0]/self.width)*self.newSize),int((bb[1]/self.height)*self.newSize),int((bb[2]/self.width)*self.newSize),int((bb[3]/self.height)*self.newSize)])
        bb_torch = torch.from_numpy(bb_np).float()
        return bb_torch
    
    
if __name__=='__main__':
    directory = '../visual_tracker_benchmark/face/'
    trainingList = ['Girl','Girl3','Girl2','Jumping','DragonBaby']
    trainingData = dict()
    numSamples = len(trainingList)
    newSize = 300
    preload = [True,'/mnt/cloudNAS2/constantine/ConvLSTM_300size_s1loss/','secondConvLSTM_250epochs_300size_2019-06-06_20:05.pth',250]
    #for data in trainingList:
    #    trainingData[data] = getVideoFrames(directory+data+'/img/',newSize)
    #img = trainingData['Girl2'][0][0]
    #gt = trainingData['Girl2'][1][0,:]
    #new = cv2.rectangle(img,(int(gt[0]),int(gt[1])),(int(gt[0])+int(gt[2]),int(gt[1])+int(gt[3])),(0,255,0),5)
    #cv2.imwrite('./test.jpg',new)
    loadData = getData(directory,[],newSize)
    numEpochs = 250
    lstm = myConvLSTM((newSize,newSize),3,3,3) #TODO: Different hidden_dim channel???
    if preload[0]:
        print('preloading weights')
        lstm.load_state_dict(torch.load(preload[1]+preload[2]))
        lstm.eval()
    lstm = lstm.cuda()
    #lr = 1e-2
    #optimizer = optim.Adadelta(lstm.parameters(), lr=lr, weight_decay=1e-05)
    optimizer = optim.Adam(lstm.parameters())
    for i in range(numEpochs):
        print(i+preload[3])
        avgLoss = 0
        for key in trainingList: #TODO: make possible to work with batch size >1
            #Prepare video sequence for going into net (unsqueeze for batch_sz=1, cuda, Variable -NO LONGER NEEDED)
            loadData.setVid(key)
            #print("Before forward pass: ",torch.cuda.memory_allocated(torch.cuda.current_device()))
            bbs = lstm(loadData,loadData.getFirstBox()) #TODO: first box as input means should really start on second frame???

            optimizer.zero_grad()
            seq_size = bbs.shape[0]
            for t in range(seq_size):
                loss =  F.smooth_l1_loss(bbs[t,:],loadData.getNextBox(),reduction='mean')#F.mse_loss(bbs[t,:],trainingData[data][1][t,:],reduction='mean')#TODO: Best loss function???
                #loss =  F.mse_loss(bbs[t,:],loadData.getNextBox(),reduction='mean')
                #loss = jaccard_loss(torch.unsqueeze(bbs[t,:]),trainingData[data][1][t,:])

                loss.backward(retain_graph=True)
            optimizer.step()
            print(torch.mean(loss,0))
            avgLoss += loss.data.item()
            del loss
            del bbs
        print("Epoch loss: ", avgLoss/numSamples)
        if i % 5 == 0 :
            torch.save(lstm.state_dict(), '../interConvLSTM_' + 'epoch'+str(i)+'.pth')

    import datetime
    now=datetime.datetime.now()
    torch.save(lstm.state_dict(), '../secondConvLSTM_' + str(numEpochs+preload[3])+ 'epochs_' + str(newSize) + 'size_' + now.strftime("%Y-%m-%d_%H:%M")+'.pth')
            
    #TODO: save optimizer state dict for further training

        
