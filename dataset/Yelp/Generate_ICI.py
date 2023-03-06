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

########################item distance matrix###############################
ItemDistance_mat = sp.dok_matrix((itemNum, itemNum))
#ITI
for i in tqdm(range(itemNum)):
    itemType = np.where(categoryMat[i].toarray()!=0)[1]#categoryMat[i,0] #type id
    for j in itemType:
        itemList = categoryDict[j]
        itemList = np.array(itemList)
        itemList2 = np.random.choice(itemList, size=int(itemList.size * 0.0015), replace=False)#
        itemList2 = itemList2.tolist()
        tmp = [i]*len(itemList2)
        ItemDistance_mat[tmp, itemList2] = 2.0
        ItemDistance_mat[itemList2, tmp] = 2.0
##final result
ItemDistance_mat = (ItemDistance_mat + sp.eye(itemNum)).tocsr()
with open('./ICI1.pkl', 'wb') as fs:
    pickle.dump(ItemDistance_mat, fs)
print(datetime.datetime.now())
print("Done")