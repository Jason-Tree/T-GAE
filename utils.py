import scipy
from graphMatching import *
from subgraphMatching import *

def load_adj(dataset):
    if (dataset == "celegans"):
        S = torch.load("data/celegans.pt")
    elif(dataset == "arenas"):
        S = torch.load("data/arenas.pt")
    elif (dataset == "douban"):
        S = torch.load("data/douban.pt")
    elif(dataset == "Online"):
        S = torch.load("data/online.pt")
    elif(dataset == "Offline"):
        S = torch.load("data/offline.pt")
    elif (dataset == "ACM"):
        S = torch.load("data/ACM.pt")
    elif (dataset == "DBLP"):
        S = torch.load("data/DBLP.pt")
    else:
        filepath = "data/" + dataset + ".npz"
        loader = load_npz(filepath)
        data = loader["adj_matrix"]
        samples = data.shape[0]
        features = data.shape[1]
        values = data.data
        coo_data = data.tocoo()
        indices = torch.LongTensor([coo_data.row, coo_data.col])
        S = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features]).to_dense()
        if (not torch.all(S.transpose(0, 1) == S)):
            S = torch.add(S, S.transpose(0, 1))
        S = S.int()
        ones = torch.ones_like(S)
        S = torch.where(S > 1, ones, S)
    return S



def test_matching(TGAE, S_hat_samples, p_samples, S_hat_features, S_emb, device, algorithm, metric):
    if (metric == "accuracy"):
        results = []
    else:
        results = {}
        results["hit@1"] = []
        results["hit@5"] = []
        results["hit@10"] = []
        results["hit@50"] = []
    for i in range(len(S_hat_samples)):
        S_hat_cur = S_hat_samples[i]
        adj = coo_matrix(S_hat_cur.numpy())
        adj_norm = preprocess_graph(adj)
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                            torch.FloatTensor(adj_norm[1]),
                                            torch.Size(adj_norm[2])).to(device)
        initial_feature = S_hat_features[i].to(device)
        z = TGAE(initial_feature, adj_norm).detach()
        D = torch.cdist(S_emb, z, p=2)
        if (metric == "accuracy"):
            if(algorithm == "greedy"):
                P_HG = greedy_hungarian(D, device)
            elif(algorithm == "exact"):
                P_HG = hungarian(D)
            elif(algorithm == "approxNN"):
                P_HG = approximate_NN(S_emb,z)
            else:
                print("Matching algorithm undefined")
                exit()
            c = 0
            P = p_samples[i]
            for j in range(P_HG.size(0)):
                r1 = P_HG[j].cpu()
                r2 = P[j].cpu()
                if (r1.equal(r2)): c += 1
            results.append(c / S_emb.shape[0])
        else:
            P = p_samples[i].T
            hitAtOne = 0
            hitAtFive = 0
            hitAtTen = 0
            hitAtFifty = 0
            for j in range(P.size(0)):
                label = torch.nonzero(P)[j][1]
                dist_list = D[j]
                sorted_neighbors = torch.argsort(dist_list).cpu()
                for hit in range(50):
                    if (sorted_neighbors[hit].item() == label):
                        if (hit == 0):
                            hitAtOne += 1
                            hitAtFive += 1
                            hitAtTen += 1
                            hitAtFifty += 1
                            break
                        elif (hit <= 4):
                            hitAtFive += 1
                            hitAtTen += 1
                            hitAtFifty += 1
                            break
                        elif (hit <= 9):
                            hitAtTen += 1
                            hitAtFifty += 1
                            break
                        elif (hit <= 49):
                            hitAtFifty += 1
                            break
            results["hit@1"].append(hitAtOne)
            results["hit@5"].append(hitAtFive)
            results["hit@10"].append(hitAtTen)
            results["hit@50"].append(hitAtFifty)

    if (metric == "accuracy"):
        results = np.array(results)
        avg = np.average(results)
        std = np.std(results)
        return avg, std
    else:
        hitAtOne = np.average(np.array(results["hit@1"]))
        stdAtOne = np.std(np.array(results["hit@1"]))
        hitAtFive = np.average(np.array(results["hit@5"]))
        stdAtFive = np.std(np.array(results["hit@5"]))
        hitAtTen = np.average(np.array(results["hit@10"]))
        stdAtTen = np.std(np.array(results["hit@10"]))
        hitAtFifty = np.average(np.array(results["hit@50"]))
        stdAtFifty = np.std(np.array(results["hit@50"]))
        num_nodes = S_emb.shape[0]
        print("Hit@1: ", end="")
        print(str(hitAtOne / num_nodes)[:6] + "+-" + str(stdAtOne / num_nodes)[:6])
        print("Hit@5: ", end="")
        print(str(hitAtFive / num_nodes)[:6] + "+-" + str(stdAtFive / num_nodes)[:6])
        print("Hit@10: ", end="")
        print(str(hitAtTen / num_nodes)[:6] + "+-" + str(stdAtTen / num_nodes)[:6])
        print("Hit@50: ", end="")
        print(str(hitAtFifty / num_nodes)[:6] + "+-" + str(stdAtFifty / num_nodes)[:6])
        print()

def gen_test_set(device,S, no_samples_each_level, perturbation_levels,method):
    S_hat_samples = {}
    S_prime_samples = {}
    p_samples = {}
    for level in perturbation_levels:
        S_hat_samples[str(level)] = []
        S_prime_samples[str(level)] = []
        p_samples[str(level)] = []
    for level in perturbation_levels:
        num_edges = int(torch.count_nonzero(S).item() / 2)
        total_purturbations = int(num_edges*level)
        if(method == "degree"):
            S = torch.triu(S, diagonal=0)
            ones_long = torch.ones((S.shape[0], 1)).type(torch.LongTensor)
            ones_int = torch.ones((S.shape[0], 1)).type(torch.IntTensor)
            ones_float = torch.ones((S.shape[0], 1)).type(torch.FloatTensor)
            try:
                D = S @ ones_long
            except:
                try:
                    D = S @ ones_int
                except:
                    D = S @ ones_float
            sum = torch.sum(torch.mul(D @ D.T, S))
            edge_index = S.nonzero().t().contiguous()
            edge_index = np.array(edge_index)
            prob = []
            for i in range(edge_index.shape[1]):
                d1 = edge_index[0, i]
                d2 = edge_index[1, i]
                prob.append(D[d1] * D[d2] / sum)
            prob = np.array(prob, dtype='float64')
            prob = np.squeeze(prob)
        for i in range(no_samples_each_level):
            if(method == "uniform"):
                add_edge = random.randint(0, total_purturbations)
                delete_edge = total_purturbations - add_edge
                S, S_prime, S_hat, P = gen_dataset(S.to(device), add_edge, delete_edge)
            elif(method == "degree"):
                edges_to_remove = np.random.choice(edge_index.shape[1], total_purturbations, False, prob)
                edges_remain = np.setdiff1d(np.array(range(edge_index.shape[1])), edges_to_remove)
                edges_index = edge_index[:, edges_remain]
                S_prime = torch.zeros_like(S)
                for j in range(edges_index.shape[1]):
                    n1 = edges_index[:, j][0]
                    n2 = edges_index[:, j][1]
                    S_prime[n1][n2] = 1
                    if (S_prime[n2][n1] == 0):
                        S_prime[n2][n1] = 1
                SIZE = S_prime.shape[0]
                permutator = torch.randperm(SIZE)
                S_hat = S_prime[permutator]
                S_hat = S_hat.t()[permutator].t()
                P = torch.zeros(SIZE, SIZE)
                for i in range(permutator.shape[0]):
                    P[i, permutator[i]] = 1
            else:
                print("Probability model not defined")
                exit()
            S_hat_samples[str(level)].append(S_hat)
            p_samples[str(level)].append(P)
            S_prime_samples[str(level)].append(S_prime)
    return S_hat_samples, S_prime_samples, p_samples

def generate_features(purturbated_S):
    features = []
    for S in purturbated_S:
        feature = gen_netsmile(S)
        features.append(feature)
    return features

def gen_dataset(S, NUM_TO_ADD, NUM_TO_DELETE):
    SIZE = S.shape[0]
    num_added = 0
    num_deleted = 0
    E = torch.zeros(S.shape[0], S.shape[0])
    edge_indexes = (S == 1).nonzero(as_tuple=False).cpu()
    blank_indexes = (S == 0).nonzero(as_tuple=False).cpu()
    """
    delete edges
    """
    while(num_deleted < NUM_TO_DELETE):

        delete_index = random.randint(0, edge_indexes.shape[0]-1)
        index = edge_indexes[delete_index]
        E[index[0]][index[1]] = -1
        E[index[1]][index[0]] = -1
        num_deleted += 1

    """
    add edges
    """
    while (num_added < NUM_TO_ADD):

        add_index = random.randint(0, blank_indexes.shape[0] - 1)
        index = blank_indexes[add_index]
        E[index[0]][index[1]] = 1
        E[index[1]][index[0]] = 1
        num_added += 1

    S_prime = torch.add(S.cpu(),E.cpu())
    permutator = torch.randperm(SIZE)
    S_hat = S_prime[permutator]
    S_hat = S_hat.t()[permutator].t()
    P = torch.zeros(SIZE, SIZE)
    for i in range(permutator.shape[0]):
        P[i, permutator[i]] = 1
    return S, S_prime, S_hat, P

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def generate_purturbations(device, S, perturbation_level, no_samples, method):
    purturbated_samples = []
    if(method == "uniform"):
        for i in range(no_samples):
            num_edges = int(torch.count_nonzero(S).item()/2)
            total_purturbations = int(perturbation_level * num_edges)
            add_edge = random.randint(0,total_purturbations)
            delete_edge = total_purturbations - add_edge
            S, S_prime, S_hat, P = gen_dataset(S.to(device), add_edge, delete_edge)
            purturbated_samples.append(S_prime)
    elif(method == "degree"):
        num_edges = int(torch.count_nonzero(S).item() / 2)
        total_purturbations = int(perturbation_level * num_edges)
        S = torch.triu(S, diagonal=0)
        ones_float = torch.ones((S.shape[0], 1)).type(torch.FloatTensor)
        ones_long = torch.ones((S.shape[0], 1)).type(torch.LongTensor)
        ones_int = torch.ones((S.shape[0], 1)).type(torch.IntTensor)
        try:
            D = S @ ones_long
        except:
            try:
                D = S @ ones_int
            except:
                D = S @ ones_float

        sum = torch.sum(torch.mul(D@D.T,S))
        edge_index = S.nonzero().t().contiguous()
        edge_index = np.array(edge_index)
        prob = []
        for i in range(edge_index.shape[1]):
            d1 = edge_index[0,i]
            d2 = edge_index[1,i]
            prob.append(D[d1]*D[d2]/sum)
        prob = np.array(prob,dtype='float64')
        prob = np.squeeze(prob)
        for i in range(no_samples):
            edges_to_remove = np.random.choice(edge_index.shape[1], total_purturbations,False,p=prob)
            edges_remain = np.setdiff1d(np.array(range(edge_index.shape[1])), edges_to_remove)
            edges_index = edge_index[:,edges_remain]
            S_prime = torch.zeros_like(S)
            for j in range(edges_index.shape[1]):
                n1 = edges_index[:,j][0]
                n2 = edges_index[:,j][1]
                S_prime[n1][n2] = 1
                S_prime[n2][n1] = 1
            purturbated_samples.append(S_prime)
    else:
        print("Probability model not defined.")
        exit()
    return purturbated_samples

def gen_netsmile(S):
    np_S = S.numpy()
    G = nx.from_numpy_array(np_S)
    feat = netsimile.feature_extraction(G)
    feat = torch.tensor(feat, dtype=torch.float)
    return feat

def load_npz(filepath):
    filepath = osp.abspath(osp.expanduser(filepath))
    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    if osp.isfile(filepath):
        with np.load(filepath, allow_pickle=True) as loader:
            loader = dict(loader)
            for k, v in loader.items():
                if v.dtype.kind in {'O', 'U'}:
                    loader[k] = v.tolist()

            return loader
    else:
        raise ValueError(f"{filepath} doesn't exist.")

def load_douban():
	x = loadmat("data/douban.mat")
	return (x['online_edge_label'][0][1],
			x['online_node_label'],
			x['offline_edge_label'][0][1],
			x['offline_node_label'],
			x['ground_truth'].T)
