from samplers import *
from optimizers import sgd_step,package_mxl_orders
from model import GraphModel
from preprocess import get_queue
from optimizers import calculate_f1,calculate_f1_partial


def graphsaint(args, device):

    file = open('./result/' + args.dataset + '_graphsaint_sampling.csv', 'w')
    file.write('epoch,cnt,epoch time,train loss,test loss,val f1,test f1\n')
    order_list = args.layer

    args.stop1 = 6

    graphsaint_sampler_ = graphsaint_sampler(order_list)

    susage = GraphModel(nfeat=args.dims, nhid=args.nhid, num_classes=args.num_classes,
                        layers=sum(args.layer), dropout=args.dropout, multi_class=args.multi_class,
                        order_list=order_list).to(device)

    susage.to(device)
    print(susage)

    optimizer = optim.Adam(susage.parameters())

    best_model = copy.deepcopy(susage)
    susage.zero_grad()

    q = get_queue(graphsaint_sampler_, args.dataset)

    target_nodes = np.array([], dtype=np.int64)

    ######
    # They can be commented when there is no need to use all data to obtain validation accuracy
    adj_lap, labels, feats, train_nodes, val_nodes, test_nodes = load_data(args.dataset,AI=True)
    adj_ori = adj_lap
    data_ori = torch.FloatTensor(feats).to(torch.device("cpu"))
    ######



    small_val = 0
    tmp, tmp_ = 0, 1

    stop = False
    for phase in range(100):

        if stop: break

        # Load original data in each round, we load it in advance for faster test in test process.
        # The features loader can also used in mmap mode. In this way, the codes of sampler and
        # features in current round should also be changed.
        # If there is no need, the following code can be used:
        #
        # adj_lap, labels, feats, train_nodes, val_nodes, test_nodes = load_data(args.dataset,AI=True)

        susage.eval()
        susage.zero_grad()

        time0 = time.time()

        need_num = int(len(feats) * args.Na)
        candidate_nums = int(need_num * args.alpha)

        all_num = 0
        nodes = []
        metric = []
        while all_num < candidate_nums:
            batch_data = q.pop()
            adjs = package_mxl_orders(batch_data[0], order_list, device)
            feat_data = torch.FloatTensor(feats[batch_data[1]]).to(device)
            outputs = susage.forward(feat_data, adjs)
            softmax_func = nn.Softmax(dim=1)
            soft_output = softmax_func(outputs).detach().cpu().numpy()
            met = np.sum(-np.multiply(soft_output, np.log(soft_output)), axis=1)
            nodes.append(batch_data[2])
            metric.append(met)

            all_num += len(batch_data[2])

        nodes = np.concatenate(nodes)
        metric = np.concatenate(metric)

        metric_p = metric / np.sum(metric)
        outputs = 0
        feat_data = 0

        all_nodes = nodes[np.argsort(-metric_p)]

        target_nodes = np.concatenate((target_nodes, np.random.permutation(all_nodes[:need_num])))

        graphsaint_sampler_.scale_growing(adj_lap, train_nodes, target_nodes, args.beta)

        time1 = time.time()
        round_time = time1 - time0
        print(round_time)

        # for mmap mode:
        # features = feats
        features = feats[graphsaint_sampler_.nodes]
        labs = labels[graphsaint_sampler_.nodes]

        batch_size = get_batch_size(graphsaint_sampler_.lap_matrix.shape[0] / len(feats), args.batch_size)


        best_val, cnt, epoch, flag = 0, 0, 0, True


        graphsaint_sampler_.get_ps2(batch_size)

        while True:
            epoch += 1

            t0 = time.time()
            train_data = graphsaint_sampler_.get_train_data2(args.batch_num, batch_size)
            cur_train_loss = sgd_step(susage, optimizer, features, labs, train_data, device, order_list)
            t1 = time.time()
            run_time = t1 - t0




            print('sgcn run time per epoch is %0.3f' % (t1 - t0))
            print(cur_train_loss)


            # calculate test loss
            susage.eval()
            susage.zero_grad()

            # Here uses the whole graph for faster and detailed testing. It can be simplified by
            # sampling or beforehand persistence to avoid accessing the whole dataset.
            with torch.no_grad():
                test_f1, val_f1, cur_test_loss = calculate_f1(susage, data_ori, adj_ori, labels, test_nodes,
                                                              val_nodes, 50000, device)

            if args.dataset == 'amazon':
                if val_f1 < 0.1:
                    cnt = 0
                elif val_f1 > best_val:
                    best_model = copy.deepcopy(susage)
                    best_val = val_f1
                    cnt = 0
                else:
                    cnt += 1
            else:
                if val_f1 > best_val:
                    best_model = copy.deepcopy(susage)
                    best_val = val_f1
                    cnt = 0
                else:
                    cnt += 1

            # print progress
            print('Epoch: ', epoch,
                  'cnt: ', cnt,
                  '| train loss: %.8f' % cur_train_loss,
                  '| test loss: %.8f' % cur_test_loss,
                  '| val f1: %.8f' % val_f1,
                  '| test f1: %.8f' % test_f1)

            file.write(
                '{},{},{},{},{},{},{}\n'.format(epoch, cnt, run_time, cur_train_loss, cur_test_loss,
                                                val_f1, test_f1))
            file.flush()

            if cnt > args.stop1:
                if flag:
                    if tmp < tmp_:
                        if best_val - small_val >= args.tau:
                            accuracy_gain = best_val - small_val
                            small_val = best_val
                            file.write('round time:{},accuracy gain:{}\n'.format(round_time, accuracy_gain))
                            file.flush()
                            break
                        else:
                            tmp += 1
                            accuracy_gain = best_val - small_val
                            small_val = best_val
                            file.write('round time:{},accuracy gain:{}\n'.format(round_time, accuracy_gain))
                            file.flush()
                            break
                    else:
                        if best_val - small_val >= args.tau:
                            accuracy_gain = best_val - small_val
                            small_val = best_val
                            file.write('round time:{},accuracy gain:{}\n'.format(round_time, accuracy_gain))
                            file.flush()
                            break
                        else:
                            accuracy_gain = best_val - small_val
                            small_val = best_val
                            file.write('round time:{},accuracy gain:{}\n'.format(round_time, accuracy_gain))
                            file.flush()
                            flag = False
                            continue

                else:
                    if cnt > args.stop2:
                        accuracy_gain = best_val - small_val
                        small_val = best_val
                        file.write('accuracy further gain:{}\n\n'.format(accuracy_gain))
                        file.flush()
                        stop = True
                        break

        '''
        # To get more accurate accuracy gain
        print('begin reset')
        for p in susage.parameters():
            p.data.clamp_(-0.6, 0.6)
        '''



    with torch.no_grad():
        test_f1, val_f1, cur_test_loss = calculate_f1(best_model, data_ori, adj_ori, labels, test_nodes,
                                                        val_nodes, 50000, device)


    file.write("test_f1:{},val_f1:{}\n".format(test_f1,val_f1))
    file.flush()
    print("test_f1:",test_f1,"|val_f1:",val_f1)
    file.close()

    return




