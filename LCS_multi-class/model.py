from utils import *
from layers import GraphConvolution, GraphSageConvolution,Aggregator
import autograd_wl
import torch.nn.functional as F
from optimizers import sgd_step,package_mxl_orders
##########################################
##########################################
##########################################
class Net(nn.Module):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, multi_class):
        super(Net, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.multi_class = multi_class

        self.gcs = None
        self.gc_out = None
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        if multi_class:
            self.loss_f = nn.BCEWithLogitsLoss()
            self.loss_f_vec = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_f = nn.CrossEntropyLoss()
            self.loss_f_vec = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, adjs):
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, adjs[ell])
            x = self.relu(x)
            x = self.dropout(x)
        x = self.gc_out(x)
        return x

    def partial_grad(self, x, adjs, targets, weight=None):
        outputs = self.forward(x, adjs)
        if weight is None:
            loss = self.loss_f(outputs, targets)
        else:
            if self.multi_class:
                loss = self.loss_f_vec(outputs, targets)
                loss = loss.mean(1) * weight
            else:
                loss = self.loss_f_vec(outputs, targets) * weight
            loss = loss.sum()
        loss.backward()
        return loss.detach()
    
    def partial_grad_with_norm(self, x, adjs, targets, weight):
        num_samples = targets.size(0)
        outputs = self.forward(x, adjs)
        
        if self.multi_class:
            loss = self.loss_f_vec(outputs, targets)
            loss = loss.mean(1) * weight
        else:
            loss = self.loss_f_vec(outputs, targets) * weight
        loss = loss.sum()
        loss.backward()
        grad_per_sample = autograd_wl.calculate_sample_grad(self.gc_out)
        
        grad_per_sample = grad_per_sample*(1/weight/num_samples)
        return loss.detach(), grad_per_sample.cpu().numpy()

    def calculate_sample_grad(self, x, adjs, targets, batch_nodes):
        # use smart way
        outputs = self.forward(x, adjs)
        loss = self.loss_f(outputs, targets[batch_nodes])
        loss.backward()
        grad_per_sample = autograd_wl.calculate_sample_grad(self.gc_out)

        return grad_per_sample.cpu().numpy()
    
    def calculate_loss_grad(self, x, adjs, targets, batch_nodes):
        outputs = self.forward(x, adjs)
        loss = self.loss_f(outputs[batch_nodes], targets[batch_nodes])
        loss.backward()
        return loss.detach()



    def calculate_f1(self, x, adjs, targets, batch_nodes):
        outputs = self.forward(x, adjs)
        if self.multi_class:
            outputs[outputs > 0] = 1
            outputs[outputs <= 0] = 0
        else:
            outputs = outputs.argmax(dim=1)
        return f1_score(outputs[batch_nodes].cpu().detach(), targets[batch_nodes].cpu().detach(), average="micro")
    
"""
This is a plain implementation of GCN
Used for FastGCN, LADIES
"""

class GCN(Net):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, multi_class):
        super(Net, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.multi_class = multi_class

        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        for _ in range(layers-1):
            self.gcs.append(GraphConvolution(nhid,  nhid))
        self.gc_out = nn.Linear(nhid, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.gc_out.register_forward_hook(autograd_wl.capture_activations)
        self.gc_out.register_backward_hook(autograd_wl.capture_backprops)

        if multi_class:
            self.loss_f = nn.BCEWithLogitsLoss()
            self.loss_f_vec = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_f = nn.CrossEntropyLoss()
            self.loss_f_vec = nn.CrossEntropyLoss(reduction='none')

class GraphSageGCN(Net):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, multi_class):
        super(Net, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.multi_class = multi_class

        self.gcs = nn.ModuleList()
        self.gcs.append(GraphSageConvolution(nfeat, nhid, use_lynorm=False))
        for _ in range(layers-1):
            self.gcs.append(GraphSageConvolution(2*nhid,  nhid, use_lynorm=False))
        self.gc_out = nn.Linear(2*nhid,  num_classes) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.gc_out.register_forward_hook(autograd_wl.capture_activations)
        self.gc_out.register_backward_hook(autograd_wl.capture_backprops)
        
        if multi_class:
            self.loss_f = nn.BCEWithLogitsLoss()
            self.loss_f_vec = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_f = nn.CrossEntropyLoss()
            self.loss_f_vec = nn.CrossEntropyLoss(reduction='none')


class GraphSageGCN_v2(Net):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, multi_class):
        super(Net, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.multi_class = multi_class

        self.gcs = nn.ModuleList()
        self.gcs.append(GraphSageConvolution(nfeat, nhid, use_lynorm=False))
        for _ in range(layers-2):
            self.gcs.append(GraphSageConvolution(2 * nhid, nhid, use_lynorm=False))
        self.gcs.append(GraphConvolution(2*nhid,  nhid))
        #self.gcs.append(nn.Linear(nhid, nhid))
        self.gc_out = nn.Linear( nhid, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.gc_out.register_forward_hook(autograd_wl.capture_activations)
        self.gc_out.register_backward_hook(autograd_wl.capture_backprops)

        if multi_class:
            self.loss_f = nn.BCEWithLogitsLoss()
            self.loss_f_vec = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_f = nn.CrossEntropyLoss()
            self.loss_f_vec = nn.CrossEntropyLoss(reduction='none')




class GraphSageGCN_v3(Net):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, multi_class,layer_list):
        super(Net, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.multi_class = multi_class
        self.layer_list = layer_list
        self.gcs = nn.ModuleList()

        for id,layer in enumerate(layer_list):
            if id == 0:
                if layer == 1:
                    self.gcs.append(GraphConvolution(nfeat,  nhid))
                elif layer == 2:
                    self.gcs.append(GraphSageConvolution(nfeat, nhid, use_lynorm=True))
            else:
                if layer_list[id-1] == 2:
                    if layer == 1:self.gcs.append(GraphConvolution(2 * nhid,  nhid))
                    elif layer == 2: self.gcs.append(GraphSageConvolution(2 * nhid, nhid, use_lynorm=True))
                    else: self.gcs.append(nn.Linear(2 * nhid, nhid))
                else:
                    if layer == 1:self.gcs.append(GraphConvolution(nhid,  nhid))
                    elif layer == 2: self.gcs.append(GraphSageConvolution(nhid, nhid, use_lynorm=True))
                    else: self.gcs.append(nn.Linear(nhid, nhid))

        self.gc_out = nn.Linear( nhid, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.gc_out.register_forward_hook(autograd_wl.capture_activations)
        self.gc_out.register_backward_hook(autograd_wl.capture_backprops)

        if multi_class:
            self.loss_f = nn.BCEWithLogitsLoss()
            self.loss_f_vec = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_f = nn.CrossEntropyLoss()
            self.loss_f_vec = nn.CrossEntropyLoss(reduction='none')


    def forward(self, x, adjs):
        num = 0
        for id,layer in enumerate(self.layer_list):
            if layer==0:
                x = self.gcs[id](x)
                x = self.relu(x)
                x = self.dropout(x)
            else:
                x = self.gcs[id](x, adjs[num])
                x = self.relu(x)
                x = self.dropout(x)
                num+=1


        x = self.gc_out(x)
        return x





class GraphModel(Net):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, multi_class,order_list):
        super(Net, self).__init__()
        self.nfeat = nfeat
        self.layers = layers
        self.nhid = nhid
        self.multi_class = multi_class
        self.order_list = order_list
        self.gcs = nn.ModuleList()



        self.dims = self.set_dims()





        for id,dim in enumerate( self.dims):
            self.gcs.append(Aggregator(dim[0], dim[1], dropout, act='relu', order=self.order_list[id], bias='norm'))

        self.gc_out = nn.Linear( nhid, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.gc_out.register_forward_hook(autograd_wl.capture_activations)
        self.gc_out.register_backward_hook(autograd_wl.capture_backprops)

        if multi_class:
            self.loss_f = nn.BCEWithLogitsLoss()
            self.loss_f_vec = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_f = nn.CrossEntropyLoss()
            self.loss_f_vec = nn.CrossEntropyLoss(reduction='none')


    def set_dims(self):

        dims = []

        for id,order in enumerate(self.order_list):
            if id == 0:
                dims.append((self.nfeat,self.nhid))
            else:
                if self.order_list[id-1]==1:
                    dims.append((2*self.nhid, self.nhid))
                else:dims.append((self.nhid, self.nhid))

        return dims

    def forward(self, x, adjs):



        for id, order in enumerate(self.order_list):
            # print(type(adjs[id]))
            x = self.gcs[id](adjs[id], x)
        x = F.normalize(x, p=2, dim=1)
        x = self.gc_out(x)
        return x



    def calculate_f1(self, x, adjs, targets):
        outputs = self.forward(x, adjs)
        if self.multi_class:

            outputs[outputs > 0] = 1
            outputs[outputs <= 0] = 0


        else:
            outputs = outputs.argmax(dim=1)


        return f1_score(outputs.cpu().detach(), targets.cpu().detach(),average='micro')

    def calculate_f1_2(self, x, adjs, targets, te_nodes, va_nodes):


        outputs = self.forward(x, adjs)




        loss = self.loss_f(outputs[te_nodes], targets[te_nodes])

        if self.multi_class:

            outputs[outputs > 0] = 1
            outputs[outputs <= 0] = 0
        else:
            outputs = outputs.argmax(dim=1)

        test_value = f1_score(outputs[te_nodes].cpu().detach(), targets[te_nodes].cpu().detach(), average='micro')
        val_value = f1_score(outputs[va_nodes].cpu().detach(), targets[va_nodes].cpu().detach(), average='micro')

        return test_value, val_value, loss.detach()

    def calculate_loss_grad(self, x, adjs, targets, batch_nodes):
        outputs = self.forward(x, adjs)
        loss = self.loss_f(outputs, targets[batch_nodes])
        return loss.detach()






    def calculate_loss_grad(self, x, adjs, targets, batch_nodes):
        outputs = self.forward(x, adjs)
        loss = self.loss_f(outputs, targets[batch_nodes])
        return loss.detach()