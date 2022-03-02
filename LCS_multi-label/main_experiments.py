import argparse
from preprocess import process
from utils import *
from subgraph_gcn import graphsaint
from node_wise_gcn import graphsage
from layer_wise_gcn import ladies
from history_wise_gcn import vr_gcn


torch.manual_seed(43)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




"""
Train Configs
"""
parser = argparse.ArgumentParser(
    description='Training GCN on Large-scale Graph Datasets by label-centric cumulative sampling')

parser.add_argument('--dataset', type=str, default='reddit',
                    help='Dataset name: pubmed/reddit')
parser.add_argument('--nhid', type=int, default=128,
                    help='Hidden state dimension')
parser.add_argument('--batch_num', type=int, default=20,
                    help='Batch Number per Epoch')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='size of output node in a batch')
parser.add_argument('--layer', type=list, default=[1,1,0],
                    help='Number of GCN layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate')
parser.add_argument('--cuda', type=int, default=1,
                    help='Avaiable GPU ID')
parser.add_argument('--stop1', type=int, default=4,
                    help='inner stop 1')
parser.add_argument('--stop2', type=int, default=15,
                    help='inner stop 2')
parser.add_argument('--Na', type=float, default=0.01,
                    help='anchor nodes ratio')
parser.add_argument('--alpha', type=float, default=5,
                    help='anchor nodes candidate parameter')
parser.add_argument('--beta', type=float, default=1,
                    help='support nodes parameter')
parser.add_argument('--tau', type=float, default=0.01,
                    help='stop condition')
args = parser.parse_args()
print(args)

#node_num/10000
if args.dataset == 'pubmed':args.batch_num = 2
if args.dataset == 'reddit':args.batch_num = 20





"""
Load Data
"""
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")


if not process(args.dataset):
    print('Error in dataset')
    exit()




args.multi_class = False
args.num_classes = np.load('./data/{}/processed/labels.npy'.format(args.dataset)).max().item()+1
args.dims = np.load('./data/{}/processed/features.npy'.format(args.dataset),mmap_mode = 'r').shape[1]
args.scale = np.load('./data/{}/processed/features.npy'.format(args.dataset),mmap_mode = 'r').shape[0]





print(args.num_classes,args.dims ,args.scale )


####################        MAIN

graphsaint(args,device)








