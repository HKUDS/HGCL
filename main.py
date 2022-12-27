# coding=UTF-8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dataloader  
import torch.optim as optim
import pickle
import random
import numpy as np
import time
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import os
from ToolScripts.TimeLogger import log
from ToolScripts.BPRData import BPRData  
import ToolScripts.evaluate as evaluate
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model  import MODEL
from args  import make_args


modelUTCStr = str(int(time.time()))
device_gpu = t.device("cuda")

isLoadModel = False



class Hope():
    def __init__(self, args, data, distanceMat, itemMat):
        self.args = args 
        self.userDistanceMat, self.itemDistanceMat, self.uiDistanceMat = distanceMat
        self.userMat = (self.userDistanceMat != 0) * 1
        self.itemMat = (itemMat != 0) * 1
        self.uiMat = (self.uiDistanceMat != 0) * 1  
       
        self.trainMat, testData, _, _, _ = data
        self.userNum, self.itemNum = self.trainMat.shape
        train_coo = self.trainMat.tocoo()
        train_u, train_v, train_r = train_coo.row, train_coo.col, train_coo.data
        assert np.sum(train_r == 0) == 0
        train_data = np.hstack((train_u.reshape(-1,1),train_v.reshape(-1,1))).tolist()
        test_data = testData
        train_dataset = BPRData(train_data, self.itemNum, self.trainMat, 1, True)
        test_dataset =  BPRData(test_data, self.itemNum, self.trainMat, 0, False)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=0) 
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=1024*1000, shuffle=False,num_workers=0)
        self.train_losses = []
        self.test_hr = []
        self.test_ndcg = []
    
    def prepareModel(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        self.model = MODEL(
                           self.args,
                           self.userNum,
                           self.itemNum,
                           self.userMat,self.itemMat, self.uiMat,
                           self.args.hide_dim,
                           self.args.Layers).cuda()
        self.opt = optim.Adam(self.model.parameters(), lr=self.args.lr)

    def predictModel(self,user, pos_i, neg_j, isTest=False):
        if isTest:
            pred_pos = t.sum(user * pos_i, dim=1)
            return pred_pos
        else:
            pred_pos = t.sum(user * pos_i, dim=1)
            pred_neg = t.sum(user * neg_j, dim=1)
            return pred_pos, pred_neg

    def adjust_learning_rate(self):
        if self.opt != None:
            for param_group in self.opt.param_groups:
                param_group['lr'] = max(param_group['lr'] * self.args.decay, self.args.minlr)

    def getModelName(self):
        title = "SR-HAN" + "_"
        ModelName = title + self.args.dataset + "_" + modelUTCStr +\
        "_hide_dim_" + str(self.args.hide_dim) +\
        "_lr_" + str(self.args.lr) +\
        "_reg_" + str(self.args.reg) +\
        "_topK_" + str(self.args.topk)+\
        "-ssl_ureg_" + str(self.args.ssl_ureg) +\
        "-ssl_ireg_" + str(self.args.ssl_ireg)
        return ModelName

    def saveHistory(self): 
        history = dict()
        history['loss'] = self.train_losses
        history['hr'] = self.test_hr
        history['ndcg'] = self.test_ndcg
        ModelName = self.getModelName()
        with open(r'./History/' + dataset + r'/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self): 
        ModelName = self.getModelName()
        history = dict()
        history['loss'] = self.train_losses
        history['hr'] = self.test_hr
        history['ndcg'] = self.test_ndcg
        savePath = r'./Model/' + dataset + r'/' + ModelName + r'.pth'
        params = {
            'model': self.model,
            'epoch': self.curEpoch,
            'args': self.args,
            'opt': self.opt,
            'history':history
            }
        t.save(params, savePath)
        log("save model : " + ModelName)

    def loadModel(self, modelPath):
        checkpoint = t.load(r'./Model/' + dataset + r'/' + modelPath + r'.pth')
        self.curEpoch = checkpoint['epoch'] + 1
        self.model = checkpoint['model']
        self.args = checkpoint['args']
        self.opt = checkpoint['opt']
        history = checkpoint['history']
        self.train_losses = history['loss']
        self.test_hr = history['hr']
        self.test_ndcg = history['ndcg']
        log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))
    
    # Contrastive Learning
    def ssl_loss(self, data1, data2,   index):
        index=t.unique(index)
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        norm_embeddings1 = F.normalize(embeddings1, p = 2, dim = 1)
        norm_embeddings2 = F.normalize(embeddings2, p = 2, dim = 1)
        pos_score  = t.sum(t.mul(norm_embeddings1, norm_embeddings2), dim = 1)
        all_score  = t.mm(norm_embeddings1, norm_embeddings2.T)
        pos_score  = t.exp(pos_score / self.args.ssl_temp)
        all_score  = t.sum(t.exp(all_score / self.args.ssl_temp), dim = 1)
        ssl_loss  = (-t.sum(t.log(pos_score / ((all_score))))/(len(index)))
        return ssl_loss
    
    # Model train
    def trainModel(self):
        epoch_loss = 0
        self.train_loader.dataset.ng_sample() 
        step_num = 0 # count batch num
        for user, item_i, item_j in self.train_loader:  
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            item_j = item_j.long().cuda()  
            step_num += 1
            self.train= True
            itemindex = t.unique(t.cat((item_i, item_j)))
            userindex = t.unique(user)
            self.userEmbed, self.itemEmbed, self.ui_userEmbedall, self.ui_itemEmbedall, self.ui_userEmbed, self.ui_itemEmbed, metaregloss = self.model( self.train, userindex, itemindex, norm=1)
            
            # Contrastive Learning of collaborative relations
            ssl_loss_user = self.ssl_loss(self.ui_userEmbed, self.userEmbed, user)    
            ssl_loss_item = self.ssl_loss(self.ui_itemEmbed, self.itemEmbed, item_i)
            ssl_loss = self.args.ssl_ureg * ssl_loss_user +  self.args.ssl_ireg * ssl_loss_item
            
            # prediction
            pred_pos, pred_neg = self.predictModel(self.ui_userEmbedall[user],  self.ui_itemEmbedall[item_i],  self.ui_itemEmbedall[item_j])
            bpr_loss = - nn.LogSigmoid()(pred_pos - pred_neg).sum()  
            epoch_loss += bpr_loss.item()
            regLoss = (t.norm(self.ui_userEmbedall[user])**2 + t.norm( self.ui_itemEmbedall[item_i])**2 + t.norm( self.ui_itemEmbedall[item_j])**2) 
            loss = ((bpr_loss + regLoss * self.args.reg ) / self.args.batch) + ssl_loss*self.args.ssl_beta + metaregloss*self.args.metareg
            
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),  max_norm=20, norm_type=2)
            self.opt.step()
        return epoch_loss 

    def testModel(self):
        HR=[]
        NDCG=[]
        
        with t.no_grad():
            uid = np.arange(0,self.userNum)
            iid = np.arange(0,self.itemNum)
            self.train = False
            _,_, self.ui_userEmbed, self.ui_itemEmbed,_,_,_= self.model( self.train,uid,iid,norm=1)
            for test_u, test_i in self.test_loader:
                test_u = test_u.long().cuda()
                test_i = test_i.long().cuda()
                pred = self.predictModel( self.ui_userEmbed[test_u], self.ui_itemEmbed[test_i], None, isTest=True)
                batch = int(test_u.cpu().numpy().size/100)
                for i in range(batch):
                    batch_socres=pred[i*100:(i+1)*100].view(-1)
                    _,indices=t.topk(batch_socres,self.args.topk) 
                    tmp_item_i=test_i[i*100:(i+1)*100]
                    recommends=t.take(tmp_item_i,indices).cpu().numpy().tolist()
                    gt_item=tmp_item_i[0].item()
                    HR.append(evaluate.hit(gt_item,recommends))
                    NDCG.append(evaluate.ndcg(gt_item,recommends))
        return np.mean(HR),np.mean(NDCG)
    
    def run(self):
        self.prepareModel()
        if isLoadModel:
            # self.loadModel(LOAD_MODEL_PATH)
            HR,NDCG = self.testModel()
            log("HR@10=%.4f, NDCG@10=%.4f"%(HR, NDCG))
            return 
        self.curEpoch = 0
        best_hr=-1
        best_ndcg=-1
        best_epoch=-1
        HR_lis=[]
        wait=0
        for e in range(args.epochs+1):
            self.curEpoch = e
            # train
            log("**************************************************************")
            epoch_loss  = self.trainModel()
            self.train_losses.append(epoch_loss)
            log("epoch %d/%d, epoch_loss=%.2f"%(e, args.epochs, epoch_loss))

            # test
            HR, NDCG = self.testModel()#
            self.test_hr.append(HR)
            self.test_ndcg.append(NDCG)
            log("epoch %d/%d, HR@10=%.4f, NDCG@10=%.4f"%(e, args.epochs, HR, NDCG))
            self.adjust_learning_rate()     
            if HR>best_hr:
                best_hr,best_ndcg,best_epoch = HR,NDCG,e
                wait=0
                # self.saveModel()
            else:
                wait+=1
                print('wait=%d'%(wait))
            HR_lis.append(HR)
            self.saveHistory()
            if wait==self.args.patience:
                log('Early stop! best epoch = %d'%(best_epoch))
                # self.loadModel(self.getModelName())
                break

        print("*****************************")
        log("best epoch = %d, HR= %.4f, NDCG=%.4f"% (best_epoch,best_hr,best_ndcg)) 
        print("*****************************")   
        print(self.args)
        log("model name : %s"%(self.getModelName()))
       
if __name__ == '__main__':
    # hyper parameters
    args = make_args()
    print(args)
    dataset = args.dataset

    # train & test data
    with open(r'dataset/'+args.dataset+'/data.pkl', 'rb') as fs:
        data = pickle.load(fs)
    with open(r'dataset/'+ args.dataset + '/distanceMat_addIUUI.pkl', 'rb') as fs:
        distanceMat = pickle.load(fs) 
    with open(r"dataset/" + args.dataset + "/ICI.pkl", "rb") as fs:
        itemMat = pickle.load(fs)

    # model instance
    hope = Hope(args, data, distanceMat, itemMat)
    modelName = hope.getModelName()
    print('ModelName = ' + modelName)    
    hope.run()
   

    

  

