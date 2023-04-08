import argparse

def make_args():
    parser = argparse.ArgumentParser(description='SR-HAN main.py')
    parser.add_argument('--dataset', type=str, default='Epinions')
    parser.add_argument('--batch', type = int, default=8192, metavar='N', help='input batch size for training')  
    parser.add_argument('--seed', type = int, default=29, metavar='int', help='random seed')
    parser.add_argument('--decay', type = float, default=0.97, metavar='LR_decay', help='decay')
    parser.add_argument('--lr', type = float, default=0.055, metavar='LR', help='learning rate')
    parser.add_argument('--minlr', type = float,default=0.0001)
    parser.add_argument('--reg', type = float, default=0.043) 
    parser.add_argument('--epochs', type = int, default=400, metavar='N', help='number of epochs to train')
    parser.add_argument('--patience', type = int, default=5, metavar='int', help='early stop patience')
    parser.add_argument('--topk', type = int, default=10)
    parser.add_argument('--hide_dim', type = int, default=32, metavar='N', help='embedding size')
    parser.add_argument('--layer_dim',nargs='?', default='[32]', help='Output size of every layer') 
    parser.add_argument('--Layers', type = int, default=2, help='the numbers of uu-GCN layer') 
    parser.add_argument('--rank', type = int, default=3, help='the dimension of low rank matrix decomposition') 
    # aggreation of the features of parameters
    parser.add_argument('--wu1', type = float, default = 0.8, help='the coefficient of feature fusion ') 
    parser.add_argument('--wu2', type = float, default = 0.2, help='the coefficient of feature fusion') 
    parser.add_argument('--wi1', type = float, default = 0.8, help='the coefficient of feature fusion ') 
    parser.add_argument('--wi2', type = float, default = 0.2, help='the coefficient of feature fusion') 

    parser.add_argument('--gcn_act', default='prelu',help='metaPath gcn activation function')
    parser.add_argument('--permute', type = int, default=1, help='whether permute subsets')
    parser.add_argument('--graphSampleSeed', type = int, default=5000, help='num sampled graph node')
    parser.add_argument('--metareg', type = float, default=0.15, help='weight of loss with reg') 
    # ssl loss
    parser.add_argument('--ssl_beta', type = float, default=0.32, help='weight of loss with ssl') 
    parser.add_argument('--ssl_temp', type = float, default=0.5, help='the temperature in softmax')
    parser.add_argument('--ssl_ureg', type = float, default=0.04)
    parser.add_argument('--ssl_ireg', type = float, default=0.05)
    parser.add_argument('--ssl_reg', type  = float, default=0.01)
    parser.add_argument('--ssl_uSamp', type = int, default=512)
    parser.add_argument('--ssl_iSamp', type = int, default=512)
    args = parser.parse_args()

    return args
