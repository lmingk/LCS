from utils import *
import autograd_wl
"""
Wrapper for variance reduction opts
"""

class ForwardWrapper(nn.Module):
    def __init__(self, n_nodes, n_hid, n_layers, n_classes, concat=False):
        super(ForwardWrapper, self).__init__()
        self.n_layers = n_layers
        if concat:
            self.hiddens = torch.zeros(n_layers, n_nodes, 2*n_hid)
        else:
            self.hiddens = torch.zeros(n_layers, n_nodes, n_hid)

    def forward_full(self, net, x, adjs, sampled_nodes):
        for ell in range(len(net.gcs)):
            x = net.gcs[ell](x, adjs[ell])
            self.hiddens[ell,sampled_nodes[ell]] = x.cpu().detach()
            x = net.relu(x)
            x = net.dropout(x)
            
        x = net.gc_out(x)
        return x

    def forward_mini(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes):
        cached_outputs = []
        for ell in range(len(net.gcs)):
            x_bar = x if ell == 0 else net.dropout(net.relu(self.hiddens[ell-1,sampled_nodes[ell-1]].to(x)))
            x_bar_exact = x_exact[input_exact_nodes[ell]] if ell == 0 else net.dropout(net.relu(self.hiddens[ell-1,input_exact_nodes[ell]].to(x)))
            x = net.gcs[ell](x, adjs[ell]) - net.gcs[ell](x_bar, adjs[ell]) + net.gcs[ell](x_bar_exact, adjs_exact[ell])
            cached_outputs += [x.detach().cpu()]
            x = net.relu(x)
            x = net.dropout(x)

        x = net.gc_out(x)
    
        for ell in range(len(net.gcs)):
            self.hiddens[ell, sampled_nodes[ell]] = cached_outputs[ell]
        return x
    
    def calculate_sample_grad(self, net, x, adjs, sampled_nodes, targets, batch_nodes):
        outputs = self.forward_full(net, x, adjs, sampled_nodes)
        loss = net.loss_f(outputs, targets[batch_nodes])
        loss.backward()
        grad_per_sample = autograd_wl.calculate_sample_grad(net.gc_out)
        return grad_per_sample.cpu().numpy()
        
    def partial_grad(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes, targets, weight=None):
        outputs = self.forward_mini(net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes)
        if weight is None:
            loss = net.loss_f(outputs, targets)
        else:
            if net.multi_class:
                loss = net.loss_f_vec(outputs, targets)
                loss = loss.mean(1) * weight
            else:
                loss = net.loss_f_vec(outputs, targets) * weight
            loss = loss.sum()
        loss.backward()
        return loss.detach()
    
    def partial_grad_with_norm(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes, targets, weight):
        num_samples = targets.size(0)
        outputs = self.forward_mini(net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes)
        
        if net.multi_class:
            loss = net.loss_f_vec(outputs, targets)
            loss = loss.mean(1) * weight
        else:
            loss = net.loss_f_vec(outputs, targets) * weight
        loss = loss.sum()
        loss.backward()
        grad_per_sample = autograd_wl.calculate_sample_grad(net.gc_out)
        
        grad_per_sample = grad_per_sample*(1/weight/num_samples)
        return loss.detach(), grad_per_sample.cpu().numpy()





















class ForwardWrapper_v2(nn.Module):
    def __init__(self, n_nodes, n_hid, n_layers, n_classes, concat=False):
        super(ForwardWrapper_v2, self).__init__()
        self.n_layers = n_layers
        if concat:
            self.hiddens = torch.zeros(n_layers, n_nodes, 2*n_hid)
        else:
            self.hiddens = torch.zeros(n_layers, n_nodes, n_hid)

    def forward_full(self, net, x, adjs, sampled_nodes):
        for ell in range(len(net.gcs)):
            x = net.gcs[ell](x, adjs[ell])
            self.hiddens[ell,sampled_nodes[ell]] = x.cpu().detach()
            x = net.dropout(net.relu(x))

        x = net.gc_out(x)
        return x

    def forward_mini(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes):
        cached_outputs = []
        for ell in range(len(net.gcs)):
            x_bar_exact = x_exact[input_exact_nodes[ell]] if ell == 0 else net.dropout(net.relu(self.hiddens[ell-1,input_exact_nodes[ell]].to(x)))
            x = torch.cat([x, x_bar_exact], dim=0)
            x = net.gcs[ell](x, adjs_exact[ell])
            cached_outputs += [x.detach().cpu()]
            x = net.dropout(net.relu(x))

        x = net.gc_out(x)
    
        for ell in range(len(net.gcs)):
            self.hiddens[ell, sampled_nodes[ell]] = cached_outputs[ell]
        return x
    
    def calculate_sample_grad(self, net, x, adjs, sampled_nodes, targets, batch_nodes):
        outputs = self.forward_full(net, x, adjs, sampled_nodes)
        loss = net.loss_f(outputs, targets[batch_nodes])
        loss.backward()
        grad_per_sample = autograd_wl.calculate_sample_grad(net.gc_out)
        return grad_per_sample.cpu().numpy()
        
    def partial_grad(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes, targets, weight=None):
        outputs = self.forward_mini(net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes)
        if weight is None:
            loss = net.loss_f(outputs, targets)
        else:
            if net.multi_class:
                loss = net.loss_f_vec(outputs, targets)
                loss = loss.mean(1) * weight
            else:
                loss = net.loss_f_vec(outputs, targets) * weight
            loss = loss.sum()
        loss.backward()
        return loss.detach()
    
    def partial_grad_with_norm(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes, targets, weight):
        num_samples = targets.size(0)
        outputs = self.forward_mini(net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes)
        
        if net.multi_class:
            loss = net.loss_f_vec(outputs, targets)
            loss = loss.mean(1) * weight
        else:
            loss = net.loss_f_vec(outputs, targets) * weight
        loss = loss.sum()
        loss.backward()
        grad_per_sample = autograd_wl.calculate_sample_grad(net.gc_out)
        
        grad_per_sample = grad_per_sample*(1/weight/num_samples)
        return loss.detach(), grad_per_sample.cpu().numpy()








class HistoryWrapper(nn.Module):
    def __init__(self, n_nodes, n_hid, order_list ):
        super(HistoryWrapper, self).__init__()
        self.order_list = order_list
        self.n_layer = np.sum(order_list)
        self.hiddens = []
        for i in order_list:
            if i == 1:
                self.hiddens.append(torch.zeros(n_nodes, n_hid*2))
            else:self.hiddens.append(torch.zeros(n_nodes, n_hid))

    def forward_mini(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes,device):
        cached_outputs = []
        x = torch.FloatTensor(x).to(device)
        for ell in range(len(net.gcs)):
            if ell == 0:
                x_bar = x
                x_bar_exact = torch.FloatTensor(x_exact[input_exact_nodes[ell]]).to(device)
            else:
                x_bar = self.hiddens[ell - 1][sampled_nodes[ell - 1]].to(device)
                x_bar_exact = self.hiddens[ell - 1][input_exact_nodes[ell]].to(device)

            x = net.gcs[ell](adjs[ell], x) - net.gcs[ell](adjs[ell], x_bar) + net.gcs[ell](adjs_exact[ell], x_bar_exact)
            cached_outputs += [x.detach().cpu()]

        x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = net.gc_out(x)

        for ell in range(len(net.gcs)):
            self.hiddens[ell][sampled_nodes[ell]] = cached_outputs[ell]
        return x

    def partial_grad(self, net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes, targets, weight=None,device = -1):
        outputs = self.forward_mini(net, x, adjs, sampled_nodes, x_exact, adjs_exact, input_exact_nodes,device)
        targets = torch.FloatTensor(targets).to(device)
        if weight is None:
            loss = net.loss_f(outputs, targets)
        else:
            if net.multi_class:
                loss = net.loss_f_vec(outputs, targets)
                loss = loss.mean(1) * weight
            else:
                loss = net.loss_f_vec(outputs, targets) * weight
            loss = loss.sum()
        loss.backward()
        return loss.detach()





