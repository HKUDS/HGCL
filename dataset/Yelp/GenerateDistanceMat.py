import numpy as np
import scipy.sparse as sp
import pickle
import datetime
from tqdm import tqdm
np.random.seed(30)  #random seed


with open("data.pkl", 'rb') as fs:
    data = pickle.load(fs)

(trainMat, _, trustMat, categoryMat, categoryDict) = data
trainMat=trainMat.tocsr()
userNum, itemNum = trainMat.shape

print(datetime.datetime.now())
########################user-item distance matrix###############################
UiDistance_mat = (sp.dok_matrix((userNum+itemNum, userNum+itemNum))).tocsr()
trainMat_T = trainMat.T
for i in tqdm(range(userNum+itemNum)):
    if i < userNum:
        UiDistance_mat[i, userNum:] = trainMat[i]
    else:
        UiDistance_mat[i, :userNum] = trainMat_T[i-userNum]

########################user distance matrix##############################
UserDistance_mat = sp.dok_matrix((userNum, userNum))
#UU
tmp_trustMat = trustMat.tocoo()
uidList1, uidList2 = tmp_trustMat.row, tmp_trustMat.col
UserDistance_mat[uidList1, uidList2] = 1.0
UserDistance_mat[uidList2, uidList1] = 1.0
##final result
UserDistance_mat = (UserDistance_mat + sp.eye(userNum)).tocsr()
with open('./UserDistance_mat.pkl', 'wb') as fs:
    pickle.dump(UserDistance_mat, fs)

########################item distance matrix###############################
ItemDistance_mat = sp.dok_matrix((itemNum, itemNum))
#ITI
for i in tqdm(range(itemNum)):
    itemType = np.where(categoryMat[i].toarray()!=0)[1]
    for j in itemType:
        itemList = categoryDict[j]
        itemList = np.array(itemList)
        itemList2 = np.random.choice(itemList, size=int(itemList.size * 0.0015), replace=False)#
        itemList2 = itemList2.tolist()
        tmp = [i]*len(itemList2)
        ItemDistance_mat[tmp, itemList2] = 1.0
        ItemDistance_mat[itemList2, tmp] = 1.0
##final result
ItemDistance_mat = (ItemDistance_mat + sp.eye(itemNum)).tocsr()
with open('./ItemDistance_mat.pkl', 'wb') as fs:
    pickle.dump(ItemDistance_mat, fs)
    
print(datetime.datetime.now())
#############save data################
distanceMat = (UserDistance_mat, ItemDistance_mat, UiDistance_mat)

with open('distanceMat_addIUUI.pkl', 'wb') as fs:
    pickle.dump(distanceMat, fs)
print("Done")


