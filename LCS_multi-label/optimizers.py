from utils import *
import autograd_wl


def calculate_grad_variance(net, feat_data, labels, train_nodes, adjs_full):
    net_grads = []
    for name, params in net.named_parameters():
#         if 'lynorm'in name:
#             continue
        net_grads.append(params.grad.data)
    clone_net = copy.deepcopy(net)
    clone_net.zero_grad()
    _ = clone_net.calculate_loss_grad(
        feat_data, adjs_full, labels, train_nodes)

    clone_net_grad = []
    for name, params in clone_net.named_parameters():
#         if 'lynorm'in name:
#             continue
        clone_net_grad.append(params.grad.data)
    del clone_net

    variance = 0.0
    for g1, g2 in zip(net_grads, clone_net_grad):
        variance += (g1-g2).norm(2) ** 2
    variance = torch.sqrt(variance)
    return variance


def package_mxl(mxl, device):
    return [torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device) for mx in mxl]



def package_mxl_orders(mxl,orders, device):
    result= []
    for id, mx in enumerate(mxl):
        if orders[id]==0:result.append(mx)
        else:result.append(torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device))
    return result


"""
Minimal Sampling GCN
"""
def boost_step(net, optimizer, feat_data, labels,
             train_nodes, valid_nodes,
             adjs_full, train_data, inner_loop_num, device, calculate_grad_vars=False):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    running_loss = []
    iter_num = 0.0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:
        for adjs, input_nodes, output_nodes, probs_nodes, sampled_nodes in train_data:
            adjs = package_mxl(adjs, device)
            weight = 1.0/torch.FloatTensor(probs_nodes).to(device)
            # compute current stochastic gradient
            optimizer.zero_grad()
            current_loss = net.partial_grad(
                feat_data[input_nodes], adjs, labels[output_nodes], weight)

            # only for experiment purpose to demonstrate ...
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(
                    net, feat_data, labels, train_nodes, adjs_full))

            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance






'''

for scale algorithm to learn

'''



def mini_batch_with_norm(net,x,adjs,  labels, weight,device):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    num_samples = len(labels)

    current_loss = net.partial_grad(x, adjs, labels, weight)
    grad_per_sample = autograd_wl.calculate_sample_grad(net.gc_out)

    grad_per_sample = grad_per_sample * (1 / weight / num_samples)

    loss = current_loss.cpu().detach()

    return loss, grad_per_sample














"""
Minimal Sampling GCN on the fly
"""
def boost_otf_step(net, optimizer, feat_data, labels,
             train_nodes, valid_nodes,
             adjs_full, train_data, inner_loop_num, device, calculate_grad_vars=False):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    running_loss = []
    iter_num = 0.0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:
        for adjs, input_nodes, output_nodes, probs_nodes, sampled_nodes in train_data:
            adjs = package_mxl(adjs, device)
            weight = 1.0/torch.FloatTensor(probs_nodes).to(device)
            # compute current stochastic gradient
            optimizer.zero_grad()
            current_loss, current_grad_norm = net.partial_grad_with_norm(
                feat_data[input_nodes], adjs, labels[output_nodes], weight)

            # only for experiment purpose to demonstrate ...
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(
                    net, feat_data, labels, train_nodes, adjs_full))

            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance

def variance_reduced_boost_step(net, optimizer, feat_data, labels,
             train_nodes, valid_nodes,
             adjs_full, train_data, inner_loop_num, device, wrapper, calculate_grad_vars=False):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    running_loss = []
    iter_num = 0.0

    grad_variance = []
    # Run over the train_loader
    while iter_num < inner_loop_num:
        for adjs, adjs_exact, input_nodes, output_nodes, probs_nodes, sampled_nodes, input_exact_nodes in train_data:
            
            adjs = package_mxl(adjs, device)
            adjs_exact = package_mxl(adjs_exact, device)
            
            weight = 1.0/torch.FloatTensor(probs_nodes).to(device)
            # compute current stochastic gradient
            
            optimizer.zero_grad()

            current_loss = wrapper.partial_grad(net, feat_data[input_nodes], adjs, sampled_nodes, feat_data, 
                                                adjs_exact, input_exact_nodes, labels[output_nodes], weight)

            # only for experiment purpose to demonstrate ...
            if calculate_grad_vars:
                grad_variance.append(calculate_grad_variance(
                    net, feat_data, labels, train_nodes, adjs_full))

            optimizer.step()

            # print statistics
            running_loss += [current_loss.cpu().detach()]
            iter_num += 1.0

    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss, running_loss, grad_variance

"""
GCN
"""

def sgd_step(net, optimizer, feat_data, labels, train_data, device,order_list):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    running_loss = []
    iter_num = 0.0

    # Run over the train_loader
    for adjs, input_nodes,output_nodes in train_data:
        adjs = package_mxl_orders(adjs, order_list,device)


        feats = torch.FloatTensor(feat_data[input_nodes]).to(device)
        labs = torch.LongTensor( labels[output_nodes]).to(device)



        # compute current stochastic gradient
        optimizer.zero_grad()
        current_loss = net.partial_grad(
            feats, adjs, labs)

        optimizer.step()

        # print statistics
        running_loss += [current_loss.cpu().detach()]
        iter_num += 1.0


    # calculate training loss

    train_loss = np.mean(running_loss)

    return train_loss


def variance_reduced_step(net, optimizer, feat_data, labels, train_data, device, wrapper,order_list):
    """
    Function to updated weights with a SGD backpropagation
    args : net, optimizer, train_loader, test_loader, loss function, number of inner epochs, args
    return : train_loss, test_loss, grad_norm_lb
    """
    net.train()

    running_loss = []
    iter_num = 0.0

    for adjs, adjs_exact, input_nodes, output_nodes, sampled_nodes, input_exact_nodes in train_data:

        adjs = package_mxl_orders(adjs, order_list,device)
        adjs_exact = package_mxl_orders(adjs_exact,order_list, device)
        # compute current stochastic gradient
        optimizer.zero_grad()



        current_loss = wrapper.partial_grad(net,
                                            feat_data[input_nodes], adjs, sampled_nodes, feat_data, adjs_exact,
                                            input_exact_nodes, labels[output_nodes],device = device)

        optimizer.step()

        # print statistics
        running_loss += [current_loss.cpu().detach()]
        iter_num += 1.0


    # calculate training loss
    train_loss = np.mean(running_loss)

    return train_loss
