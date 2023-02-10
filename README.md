# HGCL
 ![image](https://user-images.githubusercontent.com/63046458/218037666-720fd450-bdbd-496d-b9e6-fd382eb58073.png)
Torch version is available now!
This repository contains pyTorch code and datasets for the paper:
Mengru Chen, Chao Huang, Yong Xu, Lianghao Xia, Ronghua Luo, Wei Wei (2023). **Heterogeneous Graph Contrastive Learning for Recommendation**, *Paper in arXiv, Paper in ACM. In SIGIR'22, Madrid, Spain, July 11-15*, 2023

### Inroduction 
Heterogeneous Graph Contrastive Learning for Recommendation (HGCL) devises multi-relation graph contrastive learning model combining with personalized knowledge tansformation across views, to adaptively discriminate the difference and similarity between the collaborative relation for heterogeneous graph information networks, and effectively learn the comprehensive and high-quality nodes’ representations for recommendation predictions 
### Citation  
If you want to use our codes and datasets in your research, please cite:
@ inproceedings {hgcl2023, 
	Author   = {Chen, Mengru and 
Huang, Chao and 
Xu, Yong and
Xia, Lianghao and
Luo, Ronghua and
Wei, Wei},
   Title     = { Heterogeneous Graph Contrastive Learning for Recommendation}, 
booktitle  = {Proceedings of the 16th {ACM} international {WSDM} Conference on Web-Inspired Research involving Search and Data Mining, {SIGIR} 2023, Singapore, during February 27 to March 3, 2023.},
year      = {2023},
}
### Environment
The codes of HGCL are implemented and tested under the following development environment: 
pyTorch:
	Python=3.7.10
	Torch=1.8.1
	Numpy=1.20.3
	Scipy=1.6.2
### Datasets
We utilized three datasets to evaluate HGCL: *Yelp*,*Epinions*, and *CiaoDVD*. Following the common settings of implicit feedback, if user u_ihas rated item v_j, then the element (u_i,v_j) is set as 1, otherwise 0. We filtered out users and items with too few interactions. The datasets are divided into training set and testing set by 1: (n-1).
### How to Run the Code
Please unzip the datasets first. Also you need to create the History/ and the Models/ directories. The command to train HGCL on the Yelp/Epinions/CiaoDVD dataset is as follows. The commands specify the hyperparameter settings that generate the reported results in the paper.
* Yelp
```
python main.py --dataset Yelp --ssl_temp 0.5 --ssl_ureg 0.06 --ssl_ireg 0.07 --lr 0.058 --reg 0.05 --ssl_beta 0.45 --rank 3
```
* Epinions
```
python main.py --dataset Epinions --ssl_temp 0.5 --ssl_ureg 0.04 --ssl_ireg 0.05 --lr 0.055 --reg 0.043 --ssl_beta 0.32 --rank 3
```
* CiaoDVD
```
python main.py --dataset CiaoDVD --ssl_temp 0.6 --ssl_ureg 0.04 --ssl_ireg 0.05 --lr 0.055 --reg 0.065 --ssl_beta 0.3 --rank 3
```
### Important arguments
* `--ssl_temp` It is the temperature factor in the InfoNCE loss in our contrastive learning. The value is selected from {0.1, 0.3, 0.45, 0.5, 0.55,0.6, 0.65}.
* `--ssl_ureg, ssl_ireg` They are the weights for the contrastive learning loss of user’s and item’s aspect respectively. The value of this pair are tuned from 
{(3e-2,4e-2),( 4e-2,5e-2),( 5e-2,6e-2), (6e-2,7e-2),( 7e-2,8e-2)}.
* `--lr` The learning rate of the mode. We tuned it from
{1e-2, 3e-2, 4e-2, 4.5e-2, 5e-2, 5.5e-2, 6e-2}.
* `--Reg` It is the weight for weight-decay regularization. We tune this hyperparameter from the set {1e-2, 3e-2, 4.3e-2, 5e-2, 6e-2, 6.5e-2, 6.8e-2}.
* `--ssl_beta` This is the balance cofficient of the total contrastive loss , which is tuned from{0.2, 0.27, 0.3, 0.32, 0.4, 0.45, 0.48, 0.5}.
* `--rank` A hyperparameter of the dimension of low rank matrix decomposition, This parameter is recommended to tune from{1, 2, 3, 4, 5}.
### Experimental Results
Performance comparison of all methods on different datasets in terms of *NDCG* and *HR*
![image](https://user-images.githubusercontent.com/63046458/218047577-1c0c8769-fb85-4c5a-89bd-eec5925658b8.png)

