from utils import *

class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.reset_parameters()

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.spmm(adj, x)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)
            
            
class GraphSageConvolution(nn.Module):
    def __init__(self, n_in, n_out, use_lynorm=True, bias=True):
        super(GraphSageConvolution, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        self.reset_parameters()
        
        if use_lynorm:
            self.lynorm = nn.LayerNorm(2*n_out, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x

    def forward(self, x, adj):
        out_node_num = adj.size(0)
        x = self.linear(x)
        support = torch.spmm(adj, x)
        x = torch.cat([x[:out_node_num], support], dim=1)
        x = self.lynorm(x)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)


class SimplifiedGraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(SimplifiedGraphConvolution, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        
    def forward(self, x, adjs):
        for adj in adjs:
            x = torch.spmm(adj, x)
        return x





F_ACT = {'relu': nn.ReLU(),
         'I': lambda x:x}


class Aggregator(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0., act='relu', \
            order=1, bias='norm', **kwargs):
        super(Aggregator, self).__init__()
        self.order = order
        self.act, self.bias = F_ACT[act], bias
        self.dropout = dropout
        self.f_lin = list()
        self.f_bias = list()
        self.offset=list()
        self.scale=list()

        for o in range(self.order+1):
            self.f_lin.append(nn.Linear(dim_in, dim_out, bias=False))
            nn.init.xavier_uniform_(self.f_lin[-1].weight)
            self.f_bias.append(nn.Parameter(torch.zeros(dim_out)))
            self.offset.append(nn.Parameter(torch.zeros(dim_out)))
            self.scale.append(nn.Parameter(torch.ones(dim_out)))




        self.f_lin = nn.ModuleList(self.f_lin)
        self.f_dropout = nn.Dropout(p=self.dropout)
        self.params = nn.ParameterList(self.f_bias + self.offset + self.scale)
        self.f_bias = self.params[:1+self.order]
        self.offset = self.params[1+self.order:2+2*self.order]
        self.scale = self.params[2+2*self.order:]







    def _spmm(self, adj_norm, _feat):
        # alternative ways: use geometric.propagate or torch.mm
        return torch.sparse.mm(adj_norm, _feat)

    def _f_feat_trans(self, _feat, _id):
        feat=self.act(self.f_lin[_id](_feat) + self.f_bias[_id])
        if self.bias=='norm':
            mean=feat.mean(dim=1).view(feat.shape[0],1)
            var=feat.var(dim=1,unbiased=False).view(feat.shape[0],1)+1e-9
            feat_out=(feat-mean)*self.scale[_id]*torch.rsqrt(var)+self.offset[_id]
        else:
            feat_out=feat
        return feat_out


    def evaluate(self, adj_norm, feat_in,begin = 0):


        feat_in = self.f_dropout(feat_in)



        feat_hop = [feat_in]
        if self.order != 0:
            out_length = adj_norm.shape[0]
        else:
            out_length = feat_in.shape[0]

        for o in range(self.order):
            with torch.no_grad():
                feat_hop.append(self._spmm(adj_norm, feat_hop[-1]))


        feat_hop[0] = feat_hop[0][begin:begin + out_length, :]
        feat_partial = [self._f_feat_trans(ft, idf) for idf, ft in enumerate(feat_hop)]






        feat_out = torch.cat(feat_partial, 1)

        return feat_out

    def forward(self, adj_norm, feat_in):
        """
        Inputs:.
            adj_norm        edge-list represented adj matrix
        """

        feat_in = self.f_dropout(feat_in)


        feat_hop = [feat_in]
        if self.order != 0:
            out_length = adj_norm.shape[0]
        else:
            out_length = feat_in.shape[0]

        for o in range(self.order):
            feat_hop.append(self._spmm(adj_norm, feat_hop[-1]))

        feat_partial = [self._f_feat_trans(ft, idf) for idf, ft in enumerate(feat_hop)]


        feat_partial[0] = feat_partial[0][0:out_length, :]



        feat_out = torch.cat(feat_partial, 1)

        return feat_out


class GraphSageConvolution_2(nn.Module):
    def __init__(self, n_in, n_out, use_lynorm=True, bias=True, order  =1):
        super(GraphSageConvolution_2, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.order = order
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        self.reset_parameters()

        if use_lynorm:
            self.lynorm = nn.LayerNorm(2 * n_out, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x

    def forward(self, x, adj):
        if self.order!= 0:
            out_node_num = adj.size(0)
            x = self.linear(x)
            support = torch.spmm(adj, x)
            x = torch.cat([x[:out_node_num], support], dim=1)
            x = self.lynorm(x)
        else:x = self.linear(x)

        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)
