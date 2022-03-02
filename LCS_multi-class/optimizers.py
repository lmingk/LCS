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
        lab = torch.FloatTensor(labels[output_nodes]).to(device)

        # compute current stochastic gradient
        optimizer.zero_grad()
        current_loss = net.partial_grad(
            feats, adjs, lab)

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

def calculate_f1_2(net, x, adj, targets, te_nodes, va_nodes, batch, device):


    nums = adj.shape[0] // batch

    for id, order in enumerate(net.order_list):
        print(id,order)

        x = x.to(device)

        if order == 1:

            xs = []
            for num in range(nums):
                mx = sparse_mx_to_torch_sparse_tensor(adj[batch * num:batch * (num + 1), :])
                adj_piece = torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device)
                xs.append(get_vecs(net, id, x, adj_piece, begin=batch * num))

            mx = sparse_mx_to_torch_sparse_tensor(adj[batch * nums:, :])
            adj_piece = torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device)
            xs.append(get_vecs(net, id, x, adj_piece, begin=batch * nums))
            x = torch.cat(xs, 0)
            x = x.data.cpu()

        else:

            xs = []
            for num in range(nums):
                mx = x[batch * num:batch * (num + 1), :]
                xs.append(get_vecs(net,id, mx,0))
            mx = x[batch * nums:, :]
            xs.append(get_vecs(net, id, mx, 0))
            x = 0
            x = torch.cat(xs, 0).data.to(device)



    x = F.normalize(x, p=2, dim=1).data
    outputs = net.gc_out(x).data




    labels = torch.FloatTensor(targets[te_nodes]).to(device)

    loss = net.loss_f(outputs[te_nodes], labels).cpu().detach().numpy()

    if net.multi_class:

        outputs[outputs > 0] = 1
        outputs[outputs <= 0] = 0


    else:
        outputs = outputs.argmax(dim=1)

    test_value = f1_score(outputs[te_nodes].cpu().detach(), targets[te_nodes].cpu().detach(), average='micro')
    val_value = f1_score(outputs[va_nodes].cpu().detach(), targets[va_nodes].cpu().detach(), average='micro')

    return test_value, val_value, loss




def get_vecs(net,id,x,adj,begin = 0):
    return net.gcs[id].evaluate(adj, x,begin=begin).data.cpu()





def calculate_f1_partial(net, x, adjs, targets, batch, device):


    for id, order in enumerate(net.order_list):
        print(id,order)

        x = x.to(device)



        if order == 1:

            adj = adjs[id]
            nums = adj.shape[0]//batch
            xs = []
            for num in range(nums):
                print(num)
                mx = sparse_mx_to_torch_sparse_tensor(adj[batch * num:batch * (num + 1), :])
                adj_piece = torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device)
                xs.append(get_vecs(net, id, x, adj_piece, begin=batch * num))

            mx = sparse_mx_to_torch_sparse_tensor(adj[batch * nums:, :])
            adj_piece = torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device)
            xs.append(get_vecs(net, id, x, adj_piece, begin=batch * nums))
            x = torch.cat(xs, 0)
            x = x.data.cpu()
        else:

            xs = []
            nums = x.shape[0]//batch
            for num in range(nums):
                print(num)
                mx = x[batch * num:batch * (num + 1), :]
                xs.append(get_vecs(net,id, mx,0))
            mx = x[batch * nums:, :]
            xs.append(get_vecs(net, id, mx, 0))
            x = 0
            x = torch.cat(xs, 0).data.to(device)

    x = F.normalize(x, p=2, dim=1).data
    outputs = net.gc_out(x).data
    targets = torch.FloatTensor(targets).to(device)

    loss = net.loss_f(outputs, targets).cpu().detach().numpy()

    if net.multi_class:

        outputs[outputs > 0] = 1
        outputs[outputs <= 0] = 0


    else:
        outputs = outputs.argmax(dim=1)

    test_value = f1_score(outputs.cpu().detach(), targets.cpu().detach(), average='micro')

    return test_value, test_value, loss





def calculate_f1(net, x, adj, targets, te_nodes, va_nodes, batch, device):


    nums = adj.shape[0] // batch

    for id, order in enumerate(net.order_list):
        print(id,order)

        x = x.to(device)

        with torch.no_grad():
            if order == 1:

                xs = []
                for num in range(nums):
                    if num%10 ==0: print(num)
                    mx = sparse_mx_to_torch_sparse_tensor(adj[batch * num:batch * (num + 1), :])
                    adj_piece = torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device)
                    xs.append(get_vecs(net, id, x, adj_piece, begin=batch * num))

                mx = sparse_mx_to_torch_sparse_tensor(adj[batch * nums:, :])
                adj_piece = torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device)
                xs.append(get_vecs(net, id, x, adj_piece, begin=batch * nums))
                x = torch.cat(xs, 0)
                x = x.data.cpu()

            else:

                xs = []
                for num in range(nums):
                    if num % 10 == 0: print(num)
                    mx = x[batch * num:batch * (num + 1), :]
                    xs.append(get_vecs(net, id, mx, 0))
                mx = x[batch * nums:, :]
                xs.append(get_vecs(net, id, mx, 0))
                x = 0


    xss = []
    for x in xs:
        mx = F.normalize(x.to(device), p=2, dim=1).data
        mx = net.gc_out(mx).data.cpu()
        xss.append(mx)



    outputs = torch.cat(xss, 0).data.to(device)


    te_labels = torch.FloatTensor(targets[te_nodes]).to(device)
    va_labels = torch.FloatTensor(targets[va_nodes]).to(device)

    loss = net.loss_f(outputs[te_nodes], te_labels).cpu().detach().numpy()

    if net.multi_class:

        outputs[outputs > 0] = 1
        outputs[outputs <= 0] = 0


    else:
        outputs = outputs.argmax(dim=1)

    test_value = f1_score(outputs[te_nodes].cpu().detach(), te_labels.cpu().detach(), average='micro')
    val_value = f1_score(outputs[va_nodes].cpu().detach(), va_labels.cpu().detach(), average='micro')

    return test_value, val_value, loss









