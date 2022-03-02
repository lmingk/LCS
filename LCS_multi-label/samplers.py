from utils import *
from collections import deque
from collections import defaultdict










class ladies_sampler():
    def __init__(self, order_list):

        self.order_list = order_list
        self.n_layer = np.sum(order_list)
        self.order_index = np.where(self.order_list)[0]

        self.train_nodes = np.array([], dtype=np.int64)
        self.lap_matrix = None
        self.nodes = None


    def scale_growing(self, adj_lap,original_train_nodes,labels,train_nodes, rate, args):

        self.train_nodes = train_nodes

        support_nodes_num = int(len(train_nodes) * rate)

        p = (self.n_layer - 1) * np.add(np.array(np.sum(adj_lap, axis=0))[0],
                                        np.array(np.sum(adj_lap.T, axis=0))[0])
        p += np.array(np.sum(adj_lap[train_nodes, :], axis=0))[0]
        ps = support_nodes_num * p / np.sum(p)

        support_mask = np.random.uniform(0, 1, adj_lap.shape[0]) <= ps
        support_nodes = np.where(support_mask)[0]

        temp_array = adj_lap[support_nodes, :]
        temp_array.data = np.power(temp_array.data, 2)

        temp = np.array(np.sum(temp_array, axis=0))[0]

        direct_connections = np.where(temp)[0]
        inter = np.intersect1d(direct_connections, train_nodes)
        direct_connections = np.intersect1d(original_train_nodes, direct_connections)
        direct_connections = np.setdiff1d(direct_connections, inter)

        index = direct_connections[np.argsort(-temp[direct_connections])]
        candidates = np.concatenate((inter, index))

        candidate_labels = labels[candidates]
        need_num = len(train_nodes)
        num_per_class = need_num // args.num_classes + 1
        select_nodes = []
        for cla in range(args.num_classes):
            select_nodes.extend(candidates[np.where(candidate_labels == cla)[0][:num_per_class]])
        self.train_nodes = np.sort(select_nodes)

        #For better accuracy
        #all_nodes = np.union1d(self.train_nodes,support_nodes)
        #self.train_nodes = np.sort(np.intersect1d(original_train_nodes,all_nodes))



        self.nodes = np.concatenate((self.train_nodes, np.setdiff1d(support_nodes, self.train_nodes)))
        adj_matrix = adj_lap[self.nodes,:][:,self.nodes].multiply(1 / ps[self.nodes])
        self.lap_matrix = row_normalize(adj_matrix)









    def mini_batch_ld(self, seed, batch_nodes):
        np.random.seed(seed)
        previous_nodes = batch_nodes
        adjs = [0] * len(self.order_list)
        for d in range(self.n_layer):

            U = self.lap_matrix[previous_nodes, :]


            l = []

            for i in previous_nodes:
                nodes = self.lap_matrix.getrow(i).indices
                l.extend(nodes)

            neighbors = np.unique(np.array(l))
            neighbors.sort()

            temp_array = self.lap_matrix[previous_nodes, :][:, neighbors]
            temp_array.data = np.power(temp_array.data, 2)

            pi = np.array(np.sum(temp_array, axis=0))[0]
            p = pi / np.sum(pi)
            s_num = np.min([len(neighbors), 2*len(batch_nodes)])
            after_nodes = np.random.choice(
                neighbors, s_num, p=p, replace=False)
            after_nodes = np.unique(after_nodes)


            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, batch_nodes)])


            ps = np.ones(self.lap_matrix.shape[0])
            ps[neighbors] = p

            adj = U[:, after_nodes].multiply(1 / ps[after_nodes])
            adj = row_normalize(adj)

            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes

        return adjs,previous_nodes, batch_nodes
        # for mmap mode:
        # return adjs,self.nodes[previous_nodes], batch_nodes





    def target_nodes_batch1(self,adj_lap,batch_nodes):

        #Approximated  by ladies

        previous_nodes = batch_nodes
        sampled_nodes = []
        adjs = [0] * len(self.order_list)
        num_nodes = adj_lap.shape[0]
        for d in range(self.n_layer):
            U = adj_lap[previous_nodes, :]

            temp_array = adj_lap[previous_nodes, :]
            temp_array.data = np.power(temp_array.data, 2)

            pi = np.array(np.sum(temp_array, axis=0))[0]
            p = pi / np.sum(pi)
            s_num = np.min([np.sum(p > 0), int(1 * len(batch_nodes))])
            after_nodes = np.random.choice(
                num_nodes, s_num, p=p, replace=False)
            after_nodes = np.unique(after_nodes)

            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])

            pss = p[after_nodes]
            pss[np.where(pss == 0)[0]] = 1
            adj = U[:, after_nodes].multiply(1 / pss)
            adj = row_normalize(adj)

            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes

        return adjs, previous_nodes




    def target_nodes_batch2(self, adj_lap, batch_nodes):

        # Approximated  by graphsage

        previous_nodes = batch_nodes
        adjs = [0] * len(self.order_list)

        for d in range(self.n_layer):
            U = adj_lap[previous_nodes, :]

            after_nodes = []
            for U_row in U:
                indices = U_row.indices
                if len(indices) != 0:
                    sampled_indices = np.random.choice(
                        indices, 5, replace=True)
                    after_nodes.append(sampled_indices)

            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adj = row_normalize(adj)

            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes

        return adjs, previous_nodes



    def target_nodes_batch3(self, adj_lap, batch_nodes):

        # Original

        previous_nodes = batch_nodes
        adjs = [0] * len(self.order_list)

        for d in range(self.n_layer):
            U = adj_lap[previous_nodes, :]
            after_nodes = np.where(np.array(np.sum(adj_lap[previous_nodes,:], axis=0))[0])[0]
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adj = row_normalize(adj)
            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes
        return adjs, previous_nodes









    def get_one_layer(self,seed,batch_nodes):
        np.random.seed(seed)
        previous_nodes = batch_nodes

        U = self.lap_matrix[previous_nodes, :]
        is_neighbor = np.array(np.sum(U, axis=0))[0] > 0
        neighbors = np.arange(len(is_neighbor))[is_neighbor]
        neighbors = np.concatenate(
            [previous_nodes, np.setdiff1d(neighbors, previous_nodes)])

        adj = U[:, neighbors]
        adj=sparse_mx_to_torch_sparse_tensor(adj)

        return adj, neighbors, previous_nodes

    def get_one_layer2(self,seed,batch_nodes):
        np.random.seed(seed)
        previous_nodes = batch_nodes

        U = self.lap_matrix[previous_nodes, :]
        is_neighbor = np.array(np.sum(U, axis=0))[0] > 0
        neighbors = np.arange(len(is_neighbor))[is_neighbor]
        neighbors = np.concatenate(
            [previous_nodes, np.setdiff1d(neighbors, previous_nodes)])

        adj = U[:, previous_nodes]
        adj=sparse_mx_to_torch_sparse_tensor(adj)

        return adj, batch_nodes, batch_nodes



    def get_train_data(self, num, batch_num):
        train_data = []

        train_nodes_p = np.ones_like(self.train_nodes) / len(self.train_nodes) * batch_num
        for iter in np.arange(num):
            num_train_nodes = len(self.train_nodes)
            sample_mask = np.random.uniform(0, 1, num_train_nodes) <= train_nodes_p
            batch_nodes = np.where(sample_mask)[0]

            data = self.mini_batch_ld(np.random.randint(2 ** 16 - 1), batch_nodes)

            train_data.append(data)
        return train_data

    def get_test(self, adj_lap,targets):

        adjs = [0] * len(self.order_list)

        previous_nodes = targets

        for layer in range(self.n_layer):
            U = adj_lap[previous_nodes, :]

            is_neighbor = np.array(np.sum(U, axis=0))[0] > 0
            neighbors = np.arange(len(is_neighbor))[is_neighbor]
            neighbors = np.concatenate(
                [previous_nodes, np.setdiff1d(neighbors, previous_nodes)])

            adj = U[:, neighbors]
            adjs[self.order_index[-layer - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = neighbors

        input_nodes = previous_nodes
        return adjs, input_nodes












#######################################


class vrgcn_sampler():
    def __init__(self,  order_list):

        self.order_list = order_list
        self.n_layer = np.sum(order_list)
        self.order_index = np.where(self.order_list)[0]

        self.train_nodes = np.array([], dtype=np.int64)
        self.lap_matrix = None
        self.nodes = None

    def mini_batch(self, seed, batch_nodes, samp_num_list):
        np.random.seed(seed)
        sampled_nodes = []
        exact_input_nodes = []
        previous_nodes = batch_nodes
        adjs = []
        adjs_exact = []
        index = 0


        for d in range(len(self.order_list)):
            if self.order_list[-d-1] == 0:
                adjs+=[0]
                adjs_exact+=[0]
                sampled_nodes.append(previous_nodes)
                exact_input_nodes.append(previous_nodes)
            else:
                U = self.lap_matrix[previous_nodes, :]
                after_nodes = []
                after_nodes_exact = []
                for U_row in U:
                    indices = U_row.indices
                    s_num = min(len(indices), samp_num_list[index])
                    sampled_indices = np.random.choice(
                        indices, s_num, replace=False)
                    after_nodes.append(sampled_indices)
                    after_nodes_exact.append(indices)
                after_nodes = np.unique(np.concatenate(after_nodes))
                after_nodes = np.concatenate(
                    [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
                after_nodes_exact = np.unique(np.concatenate(after_nodes_exact))
                after_nodes_exact = np.concatenate(
                    [previous_nodes, np.setdiff1d(after_nodes_exact, previous_nodes)])
                adj = U[:, after_nodes]
                adj = row_normalize(adj)
                adjs += [sparse_mx_to_torch_sparse_tensor(adj)]

                adj_exact = U[:, after_nodes_exact]
                adjs_exact += [sparse_mx_to_torch_sparse_tensor(adj_exact)]
                sampled_nodes.append(previous_nodes)
                exact_input_nodes.append(after_nodes_exact)
                previous_nodes = after_nodes
                index+=1


        adjs.reverse()
        sampled_nodes.reverse()
        adjs_exact.reverse()
        exact_input_nodes.reverse()
        #print(sampled_nodes)
        return adjs, adjs_exact, previous_nodes, batch_nodes, sampled_nodes, exact_input_nodes
        # for mmap mode:
        # exact_input_nodes[0] = self.nodes[exact_input_nodes[0]]
        # return adjs, adjs_exact, self.nodes[previous_nodes], self.nodes[batch_nodes], sampled_nodes, exact_input_nodes



    def scale_growing(self, adj_lap,original_train_nodes,labels,train_nodes, rate, args):  # 标签矫正

        self.train_nodes = train_nodes

        support_nodes_num = int(len(train_nodes) * rate)

        p = (self.n_layer - 1) * np.add(np.array(np.sum(adj_lap, axis=0))[0],
                                        np.array(np.sum(adj_lap.T, axis=0))[0])
        p += np.array(np.sum(adj_lap[train_nodes, :], axis=0))[0]
        ps = support_nodes_num * p / np.sum(p)

        support_mask = np.random.uniform(0, 1, adj_lap.shape[0]) <= ps
        support_nodes = np.where(support_mask)[0]


        temp_array = adj_lap[support_nodes, :]
        temp_array.data = np.power(temp_array.data, 2)

        temp = np.array(np.sum(temp_array, axis=0))[0]

        direct_connections = np.where(temp)[0]
        inter = np.intersect1d(direct_connections, train_nodes)
        direct_connections = np.intersect1d(original_train_nodes, direct_connections)
        direct_connections = np.setdiff1d(direct_connections, inter)

        index = direct_connections[np.argsort(-temp[direct_connections])]
        candidates = np.concatenate((inter, index))

        candidate_labels = labels[candidates]
        need_num = len(train_nodes)
        num_per_class = need_num // args.num_classes + 1
        select_nodes = []
        for cla in range(args.num_classes):
            select_nodes.extend(candidates[np.where(candidate_labels == cla)[0][:num_per_class]])
        self.train_nodes = np.sort(select_nodes)

        #For better accuracy
        #all_nodes = np.union1d(self.train_nodes,support_nodes)
        #self.train_nodes = np.sort(np.intersect1d(original_train_nodes,all_nodes))



        self.nodes = np.concatenate((self.train_nodes, np.setdiff1d(support_nodes, self.train_nodes)))
        adj_matrix = adj_lap[self.nodes,:][:,self.nodes].multiply(1 / ps[self.nodes])
        self.lap_matrix = row_normalize(adj_matrix)

    def target_nodes_batch1(self,adj_lap,batch_nodes):

        #Approximated  by ladies

        previous_nodes = batch_nodes
        sampled_nodes = []
        adjs = [0] * len(self.order_list)
        num_nodes = adj_lap.shape[0]
        for d in range(self.n_layer):
            U = adj_lap[previous_nodes, :]

            temp_array = adj_lap[previous_nodes, :]
            temp_array.data = np.power(temp_array.data, 2)

            pi = np.array(np.sum(temp_array, axis=0))[0]
            p = pi / np.sum(pi)
            s_num = np.min([np.sum(p > 0), int(1 * len(batch_nodes))])
            after_nodes = np.random.choice(
                num_nodes, s_num, p=p, replace=False)
            after_nodes = np.unique(after_nodes)

            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])

            pss = p[after_nodes]
            pss[np.where(pss == 0)[0]] = 1
            adj = U[:, after_nodes].multiply(1 / pss)
            adj = row_normalize(adj)

            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes

        return adjs, previous_nodes




    def target_nodes_batch2(self, adj_lap, batch_nodes):

        # Approximated  by graphsage

        previous_nodes = batch_nodes
        adjs = [0] * len(self.order_list)

        for d in range(self.n_layer):
            U = adj_lap[previous_nodes, :]

            after_nodes = []
            for U_row in U:
                indices = U_row.indices
                if len(indices) != 0:
                    sampled_indices = np.random.choice(
                        indices, 5, replace=True)
                    after_nodes.append(sampled_indices)

            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adj = row_normalize(adj)

            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes

        return adjs, previous_nodes



    def target_nodes_batch3(self, adj_lap, batch_nodes):

        # Original

        previous_nodes = batch_nodes
        adjs = [0] * len(self.order_list)

        for d in range(self.n_layer):
            U = adj_lap[previous_nodes, :]
            after_nodes = np.where(np.array(np.sum(adj_lap[previous_nodes,:], axis=0))[0])[0]
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adj = row_normalize(adj)
            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes
        return adjs, previous_nodes








    def get_test(self, adj_lap,targets):

        adjs = [0] * len(self.order_list)

        previous_nodes = targets

        for layer in range(self.n_layer):
            U = adj_lap[previous_nodes, :]

            is_neighbor = np.array(np.sum(U, axis=0))[0] > 0
            neighbors = np.arange(len(is_neighbor))[is_neighbor]
            neighbors = np.concatenate(
                [previous_nodes, np.setdiff1d(neighbors, previous_nodes)])

            adj = U[:, neighbors]
            adjs[self.order_index[-layer - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = neighbors

        input_nodes = previous_nodes
        return adjs, input_nodes



    def get_train_data(self,num,batch_num,samp_num_list):
        train_data = []
        train_nodes_p = np.ones_like(self.train_nodes)/len(self.train_nodes)*batch_num
        for iter in np.arange(num):
            num_train_nodes = len(self.train_nodes)
            sample_mask = np.random.uniform(0, 1, num_train_nodes) <= train_nodes_p
            batch_nodes = np.where(sample_mask)[0]
            data = self.mini_batch(np.random.randint(2 ** 16 - 1), batch_nodes,samp_num_list)

            train_data.append(data)
        return train_data




    def get_one_layer(self,seed,batch_nodes):
        np.random.seed(seed)
        previous_nodes = batch_nodes

        U = self.lap_matrix[previous_nodes, :]
        is_neighbor = np.array(np.sum(U, axis=0))[0] > 0
        neighbors = np.arange(len(is_neighbor))[is_neighbor]
        neighbors = np.concatenate(
            [previous_nodes, np.setdiff1d(neighbors, previous_nodes)])

        adj = U[:, neighbors]
        adj=sparse_mx_to_torch_sparse_tensor(adj)

        return adj, neighbors, previous_nodes

    def get_one_layer2(self,seed,batch_nodes):
        np.random.seed(seed)
        previous_nodes = batch_nodes

        U = self.lap_matrix[previous_nodes, :]
        is_neighbor = np.array(np.sum(U, axis=0))[0] > 0
        neighbors = np.arange(len(is_neighbor))[is_neighbor]
        neighbors = np.concatenate(
            [previous_nodes, np.setdiff1d(neighbors, previous_nodes)])

        adj = U[:, previous_nodes]
        adj=sparse_mx_to_torch_sparse_tensor(adj)

        return adj, batch_nodes, batch_nodes

















class graphsage_sampler():
    def __init__(self, order_list):

        self.order_list = order_list
        self.n_layer = np.sum(order_list)
        self.order_index = np.where(self.order_list)[0]

        self.train_nodes = np.array([], dtype=np.int64)
        self.lap_matrix = None
        self.nodes = None







    def mini_batch(self, seed, batch_nodes, samp_num_list):
        np.random.seed(seed)
        adjs = [0] * len(self.order_list)
        sampled_nodes = []
        previous_nodes = batch_nodes

        for d in range(self.n_layer):
            U = self.lap_matrix[previous_nodes, :]
            after_nodes = []
            for U_row in U:
                indices = U_row.indices
                if len(indices) != 0:
                    sampled_indices = np.random.choice(
                    indices, samp_num_list[d], replace=True)
                    after_nodes.append(sampled_indices)
            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adj = row_normalize(adj)

            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)

            previous_nodes = after_nodes

        return adjs, previous_nodes, batch_nodes
        # for mmap mode:
        # return adjs, self.nodes[previous_nodes], self.nodes[batch_nodes]


    def scale_growing(self, adj_lap,original_train_nodes,labels,train_nodes, rate, args):

        self.train_nodes = train_nodes

        support_nodes_num = int(len(train_nodes) * rate)

        p = (self.n_layer - 1) * np.add(np.array(np.sum(adj_lap, axis=0))[0],
                                        np.array(np.sum(adj_lap.T, axis=0))[0])
        p += np.array(np.sum(adj_lap[train_nodes, :], axis=0))[0]
        ps = support_nodes_num * p / np.sum(p)

        support_mask = np.random.uniform(0, 1, adj_lap.shape[0]) <= ps
        support_nodes = np.where(support_mask)[0]


        temp_array = adj_lap[support_nodes, :]
        temp_array.data = np.power(temp_array.data, 2)

        temp = np.array(np.sum(temp_array, axis=0))[0]

        direct_connections = np.where(temp)[0]
        inter = np.intersect1d(direct_connections, train_nodes)
        direct_connections = np.intersect1d(original_train_nodes, direct_connections)
        direct_connections = np.setdiff1d(direct_connections, inter)

        index = direct_connections[np.argsort(-temp[direct_connections])]
        candidates = np.concatenate((inter, index))

        candidate_labels = labels[candidates]
        need_num = len(train_nodes)
        num_per_class = need_num // args.num_classes + 1
        select_nodes = []
        for cla in range(args.num_classes):
            select_nodes.extend(candidates[np.where(candidate_labels == cla)[0][:num_per_class]])
        self.train_nodes = np.sort(select_nodes)

        #For better accuracy
        #all_nodes = np.union1d(self.train_nodes,support_nodes)
        #self.train_nodes = np.sort(np.intersect1d(original_train_nodes,all_nodes))



        self.nodes = np.concatenate((self.train_nodes, np.setdiff1d(support_nodes, self.train_nodes)))
        adj_matrix = adj_lap[self.nodes,:][:,self.nodes].multiply(1 / ps[self.nodes])
        self.lap_matrix = row_normalize(adj_matrix)




    def target_nodes_batch1(self,adj_lap,batch_nodes):

        #Approximated  by ladies

        previous_nodes = batch_nodes
        sampled_nodes = []
        adjs = [0] * len(self.order_list)
        num_nodes = adj_lap.shape[0]
        for d in range(self.n_layer):
            U = adj_lap[previous_nodes, :]

            temp_array = adj_lap[previous_nodes, :]
            temp_array.data = np.power(temp_array.data, 2)

            pi = np.array(np.sum(temp_array, axis=0))[0]
            p = pi / np.sum(pi)
            s_num = np.min([np.sum(p > 0), int(1 * len(batch_nodes))])
            after_nodes = np.random.choice(
                num_nodes, s_num, p=p, replace=False)
            after_nodes = np.unique(after_nodes)

            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])

            pss = p[after_nodes]
            pss[np.where(pss == 0)[0]] = 1
            adj = U[:, after_nodes].multiply(1 / pss)
            adj = row_normalize(adj)

            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes

        return adjs, previous_nodes




    def target_nodes_batch2(self, adj_lap, batch_nodes):

        # Approximated  by graphsage

        previous_nodes = batch_nodes
        adjs = [0] * len(self.order_list)

        for d in range(self.n_layer):
            U = adj_lap[previous_nodes, :]

            after_nodes = []
            for U_row in U:
                indices = U_row.indices
                if len(indices) != 0:
                    sampled_indices = np.random.choice(
                        indices, 5, replace=True)
                    after_nodes.append(sampled_indices)

            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adj = row_normalize(adj)

            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes

        return adjs, previous_nodes



    def target_nodes_batch3(self, adj_lap, batch_nodes):

        # Original

        previous_nodes = batch_nodes
        adjs = [0] * len(self.order_list)

        for d in range(self.n_layer):
            U = adj_lap[previous_nodes, :]
            after_nodes = np.where(np.array(np.sum(adj_lap[previous_nodes,:], axis=0))[0])[0]
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adj = row_normalize(adj)
            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes
        return adjs, previous_nodes







    def get_test(self, adj_lap,targets):

        adjs = [0] * len(self.order_list)

        previous_nodes = targets

        for layer in range(self.n_layer):
            U = adj_lap[previous_nodes, :]

            is_neighbor = np.array(np.sum(U, axis=0))[0] > 0
            neighbors = np.arange(len(is_neighbor))[is_neighbor]
            neighbors = np.concatenate(
                [previous_nodes, np.setdiff1d(neighbors, previous_nodes)])

            adj = U[:, neighbors]
            adjs[self.order_index[-layer - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = neighbors

        input_nodes = previous_nodes
        return adjs, input_nodes



    def get_train_data(self,num,batch_num,samp_num_list):
        train_data = []
        train_nodes_p = np.ones_like(self.train_nodes)/len(self.train_nodes)*batch_num
        for iter in np.arange(num):
            num_train_nodes = len(self.train_nodes)
            sample_mask = np.random.uniform(0, 1, num_train_nodes) <= train_nodes_p
            batch_nodes = np.where(sample_mask)[0]
            data = self.mini_batch(np.random.randint(2 ** 16 - 1), batch_nodes,samp_num_list)

            train_data.append(data)
        return train_data



    def get_one_layer(self,seed,batch_nodes):
        np.random.seed(seed)
        previous_nodes = batch_nodes

        U = self.lap_matrix[previous_nodes, :]
        is_neighbor = np.array(np.sum(U, axis=0))[0] > 0
        neighbors = np.arange(len(is_neighbor))[is_neighbor]
        neighbors = np.concatenate(
            [previous_nodes, np.setdiff1d(neighbors, previous_nodes)])

        adj = U[:, neighbors]
        adj=sparse_mx_to_torch_sparse_tensor(adj)

        return adj, neighbors, previous_nodes

    def get_one_layer2(self,seed,batch_nodes):
        np.random.seed(seed)
        previous_nodes = batch_nodes

        U = self.lap_matrix[previous_nodes, :]
        is_neighbor = np.array(np.sum(U, axis=0))[0] > 0
        neighbors = np.arange(len(is_neighbor))[is_neighbor]
        neighbors = np.concatenate(
            [previous_nodes, np.setdiff1d(neighbors, previous_nodes)])

        adj = U[:, previous_nodes]
        adj=sparse_mx_to_torch_sparse_tensor(adj)

        return adj, batch_nodes, batch_nodes


class graphsaint_sampler():
    def __init__(self, order_list):

        self.order_list = order_list
        self.n_layer = np.sum(order_list)
        self.order_index = np.where(self.order_list)[0]

        self.train_nodes = np.array([], dtype=np.int64)
        self.lap_matrix = None
        self.nodes = None
        self.ps = None



    def scale_growing(self, adj_lap,original_train_nodes,labels,train_nodes, rate, args):  

        self.train_nodes = train_nodes

        support_nodes_num = int(len(train_nodes) * rate)

        p = (self.n_layer - 1) * np.add(np.array(np.sum(adj_lap, axis=0))[0],
                                        np.array(np.sum(adj_lap.T, axis=0))[0])
        p += np.array(np.sum(adj_lap[train_nodes, :], axis=0))[0]
        ps = support_nodes_num * p / np.sum(p)

        support_mask = np.random.uniform(0, 1, adj_lap.shape[0]) <= ps
        support_nodes = np.where(support_mask)[0]


        temp_array = adj_lap[support_nodes, :]
        temp_array.data = np.power(temp_array.data, 2)

        temp = np.array(np.sum(temp_array, axis=0))[0]

        direct_connections = np.where(temp)[0]
        inter = np.intersect1d(direct_connections, train_nodes)
        direct_connections = np.intersect1d(original_train_nodes, direct_connections)
        direct_connections = np.setdiff1d(direct_connections, inter)

        index = direct_connections[np.argsort(-temp[direct_connections])]
        candidates = np.concatenate((inter, index))

        candidate_labels = labels[candidates]
        need_num = len(train_nodes)
        num_per_class = need_num // args.num_classes + 1
        select_nodes = []
        for cla in range(args.num_classes):
            select_nodes.extend(candidates[np.where(candidate_labels == cla)[0][:num_per_class]])
        self.train_nodes = np.sort(select_nodes)

        # For better accuracy
        #all_nodes = np.union1d(self.train_nodes,support_nodes)
        #self.train_nodes = np.sort(np.intersect1d(original_train_nodes,all_nodes))



        self.nodes = np.concatenate((self.train_nodes, np.setdiff1d(support_nodes, self.train_nodes)))
        adj_matrix = adj_lap[self.nodes,:][:,self.nodes].multiply(1 / ps[self.nodes])
        self.lap_matrix = row_normalize(adj_matrix)


    def target_nodes_batch1(self,adj_lap,batch_nodes):

        #Approximated  by ladies

        previous_nodes = batch_nodes
        sampled_nodes = []
        adjs = [0] * len(self.order_list)
        num_nodes = adj_lap.shape[0]
        for d in range(self.n_layer):
            U = adj_lap[previous_nodes, :]

            temp_array = adj_lap[previous_nodes, :]
            temp_array.data = np.power(temp_array.data, 2)

            pi = np.array(np.sum(temp_array, axis=0))[0]
            p = pi / np.sum(pi)
            s_num = np.min([np.sum(p > 0), int(1 * len(batch_nodes))])
            after_nodes = np.random.choice(
                num_nodes, s_num, p=p, replace=False)
            after_nodes = np.unique(after_nodes)

            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])

            pss = p[after_nodes]
            pss[np.where(pss == 0)[0]] = 1
            adj = U[:, after_nodes].multiply(1 / pss)
            adj = row_normalize(adj)

            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes

        return adjs, previous_nodes




    def target_nodes_batch2(self, adj_lap, batch_nodes):

        # Approximated  by graphsage

        previous_nodes = batch_nodes
        adjs = [0] * len(self.order_list)

        for d in range(self.n_layer):
            U = adj_lap[previous_nodes, :]

            after_nodes = []
            for U_row in U:
                indices = U_row.indices
                if len(indices) != 0:
                    sampled_indices = np.random.choice(
                        indices, 5, replace=True)
                    after_nodes.append(sampled_indices)

            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adj = row_normalize(adj)

            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes

        return adjs, previous_nodes



    def target_nodes_batch3(self, adj_lap, batch_nodes):

        # Original

        previous_nodes = batch_nodes
        adjs = [0] * len(self.order_list)

        for d in range(self.n_layer):
            U = adj_lap[previous_nodes, :]
            after_nodes = np.where(np.array(np.sum(adj_lap[previous_nodes,:], axis=0))[0])[0]
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adj = row_normalize(adj)
            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = after_nodes
        return adjs, previous_nodes






    def get_train_data(self, num, batch_num):
        train_data = []
        print(batch_num,'batch_num')
        for iter in np.arange(num):
            data = self.mini_batch(np.random.randint(2 ** 16 - 1), batch_num)
            train_data.append(data)
        return train_data

    def mini_batch(self, seed, batch_num):  # this batch sampling constrainit 128 // 1024
        np.random.seed(seed)
        adjs = [0] * len(self.order_list)
        label_nodes = []
        batch_nodes = set()
        step = 0
        while step < batch_num:  ############ original   while step < batch_num
            current = np.random.choice(len(self.train_nodes))
            if len(self.lap_matrix.getrow(current).indices) == 0: continue
            label_nodes.append(current)
            batch_nodes.add(current)
            for layer in range(self.n_layer):
                current = np.random.choice(self.lap_matrix.getrow(current).indices)
                batch_nodes.add(current)
            step += 1
        label_nodes = np.array(sorted(label_nodes))
        batch_nodes = np.array(sorted(list(batch_nodes)))
        batch_nodes = np.concatenate(
            [label_nodes, np.setdiff1d(batch_nodes, label_nodes)])
        for d in range(self.n_layer):
            if d == 0:
                adj = self.lap_matrix[label_nodes, :][:, batch_nodes]
            else:
                adj = self.lap_matrix[batch_nodes, :][:, batch_nodes]
            adj = adj.multiply(1 / self.ps[batch_nodes])
            adj = row_normalize(adj)
            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
        return adjs, batch_nodes, label_nodes
        # for mmap mode:
        # return adjs, self.nodes[batch_nodes], self.nodes[label_nodes]


    def get_train_data2(self, num, batch_num):
        train_data = []
        train_nodes_p = np.ones_like(self.train_nodes) / len(self.train_nodes) * batch_num
        for iter in np.arange(num):
            num_train_nodes = len(self.train_nodes)
            sample_mask = np.random.uniform(0, 1, num_train_nodes) <= train_nodes_p
            batch_nodes = np.where(sample_mask)[0]
            data = self.mini_batch2(np.random.randint(2 ** 16 - 1), batch_nodes)

            train_data.append(data)
        return train_data

    def mini_batch2(self, seed, batch):  # this batch sampling constrainit 128 // 1024
        # print(seed,'seeds')
        np.random.seed(seed)
        adjs = [0] * len(self.order_list)
        label_nodes = []
        batch_nodes = set()

        for root in batch:  ############ original   while step < batch_num
            if len(self.lap_matrix.getrow(root).indices) == 0: continue
            label_nodes.append(root)
            batch_nodes.add(root)
            for _ in range(1):
                current = root
                for layer in range(self.n_layer):
                    current = np.random.choice(self.lap_matrix.getrow(current).indices)
                    batch_nodes.add(current)

        label_nodes = np.array(sorted(label_nodes))
        batch_nodes = np.array(sorted(list(batch_nodes)))
        batch_nodes = np.concatenate(
            [label_nodes, np.setdiff1d(batch_nodes, label_nodes)])
        for d in range(self.n_layer):
            if d == 0:
                adj = self.lap_matrix[label_nodes, :][:, batch_nodes]
            else:
                adj = self.lap_matrix[batch_nodes, :][:, batch_nodes]
            adj = adj.multiply(1 / self.ps[batch_nodes])
            adj = row_normalize(adj)
            adjs[self.order_index[-d - 1]] = sparse_mx_to_torch_sparse_tensor(adj)

        return adjs, batch_nodes, label_nodes
        # for mmap mode:
        # return adjs, self.nodes[batch_nodes], self.nodes[label_nodes]


    def get_test(self, adj_lap,targets):

        adjs = [0] * len(self.order_list)

        previous_nodes = targets

        for layer in range(self.n_layer):
            U = adj_lap[previous_nodes, :]

            is_neighbor = np.array(np.sum(U, axis=0))[0] > 0
            neighbors = np.arange(len(is_neighbor))[is_neighbor]
            neighbors = np.concatenate(
                [previous_nodes, np.setdiff1d(neighbors, previous_nodes)])

            adj = U[:, neighbors]
            adjs[self.order_index[-layer - 1]] = sparse_mx_to_torch_sparse_tensor(adj)
            previous_nodes = neighbors

        input_nodes = previous_nodes
        return adjs, input_nodes




    def get_ps(self,graph_scale,batch_num):

        num = graph_scale*50/(batch_num*1*self.n_layer)
        node_count = np.zeros(self.lap_matrix.shape[0])

        for _ in range(num):

            nodes = set()

            train_nodes_p = np.ones_like(self.train_nodes) / len(self.train_nodes) * batch_num
            num_train_nodes = len(self.train_nodes)
            sample_mask = np.random.uniform(0, 1, num_train_nodes) <= train_nodes_p
            batch = np.where(sample_mask)[0]

            #batch = np.random.choice(len(self.train_nodes),batch_num,replace=False)

            for root in batch:
                if len(self.lap_matrix.getrow(root).indices) == 0: continue
                nodes.add(root)
                for _ in range(1):
                    current = root
                    for layer in range(self.n_layer):
                        current = np.random.choice(self.lap_matrix.getrow(current).indices)
                        nodes.add(current)
            for node in nodes: node_count[node]+=1
        node_p = node_count/num
        self.ps = node_p









    def get_ps3(self,batch_num):

        probability_adj = self.lap_matrix.astype(np.bool)
        probability_adj = row_normalize(probability_adj)

        previous = np.zeros(self.lap_matrix.shape[0])
        previous[0:len(self.train_nodes)] = 1 / len(self.train_nodes)
        previous = sp.csr_matrix(previous)
        p = np.zeros(self.lap_matrix.shape[0])
        for layer in range(self.n_layer):
            p = p + previous.A[0]
            previous = previous.dot(probability_adj)
            previous = previous / np.sum(previous)
        print(np.sum(previous))
        p = p + previous.A[0]
        p = p / 3
        inclusion_p = 1 - batch_num * (p)
        self.ps = inclusion_p

    def get_ps1(self, batch_num):

        probability_adj = self.lap_matrix

        previous = np.zeros(probability_adj.shape[0])
        previous[0:len(self.train_nodes)] = 1 / len(self.train_nodes)
        previous = sp.csr_matrix(previous)
        p = np.zeros(probability_adj.shape[0])
        for layer in range(self.n_layer):
            p = p + previous.A[0]
            previous = previous.dot(probability_adj)
            previous = previous / np.sum(previous)
        p = p + previous.A[0]
        p = p / 3
        inclusion_p = 1 - np.power(np.e, -p * 3 * batch_num)
        self.ps = inclusion_p

    def get_ps2(self, batch_num):

        probability_adj = self.lap_matrix.astype(np.bool)
        probability_adj = row_normalize(probability_adj)

        previous = np.zeros(probability_adj.shape[0])
        previous[0:len(self.train_nodes)] = 1 / len(self.train_nodes)
        previous = sp.csr_matrix(previous)
        p = np.zeros(probability_adj.shape[0])
        for layer in range(self.n_layer):
            p = p + previous.A[0]
            previous = previous.dot(probability_adj)
            previous = previous / np.sum(previous)
        p = p + previous.A[0]
        p = p / 3
        inclusion_p = 1 - np.power(np.e, -p * 3 * batch_num)
        self.ps = inclusion_p







































