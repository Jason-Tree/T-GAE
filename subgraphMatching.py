from graphMatching import *
from networkx import read_edgelist
from scipy.io import loadmat
from model import *
from utils import *

def fit_TGAE_subgraph(data, no_samples, GAE, epoch, train_loader, train_features, device, lr, test_pairs):
    best_hitAtOne = 0
    best_hitAtFive = 0
    best_hitAtTen = 0
    best_hitAtFifty = 0
    optimizer = Adam(GAE.parameters(), lr=lr,weight_decay=5e-4)
    for step in tqdm(range(epoch)):
        loss = 0
        for dataset in train_loader.keys():
            S = train_loader[dataset][0]
            initial_features = train_features[dataset]
            for i in range(len(train_loader[dataset])):
                adj_tensor = train_loader[dataset][i]
                adj = coo_matrix(adj_tensor.numpy())
                adj_norm = preprocess_graph(adj)
                pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
                norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

                adj_label = coo_matrix(S.numpy())
                adj_label = sparse_to_tuple(adj_label)

                adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                                    torch.FloatTensor(adj_norm[1]),
                                                    torch.Size(adj_norm[2])).to(device)
                adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                                    torch.FloatTensor(adj_label[1]),
                                                    torch.Size(adj_label[2])).to(device)

                initial_feature = initial_features[i].to(device)

                weight_mask = adj_label.to_dense().view(-1) == 1
                weight_tensor = torch.ones(weight_mask.size(0))
                weight_tensor[weight_mask] = pos_weight
                weight_tensor = weight_tensor.to(device)
                z = GAE(initial_feature, adj_norm)
                A_pred = torch.sigmoid(torch.matmul(z,z.t()))
                loss += norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1),
                                                           weight=weight_tensor)
        optimizer.zero_grad()
        loss = loss / no_samples
        loss.backward()
        optimizer.step()

        keys = list(train_loader.keys())
        S1 = train_loader[keys[0]][0]
        S2 = train_loader[keys[1]][0]
        adj_S1 = coo_matrix(S1.numpy())
        adj_norm_1 = preprocess_graph(adj_S1)
        adj_norm_1 = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_1[0].T),
                                              torch.FloatTensor(adj_norm_1[1]),
                                              torch.Size(adj_norm_1[2])).to(device)
        adj_S2 = coo_matrix(S2.numpy())
        adj_norm_2 = preprocess_graph(adj_S2)
        adj_norm_2 = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_2[0].T),
                                              torch.FloatTensor(adj_norm_2[1]),
                                              torch.Size(adj_norm_2[2])).to(device)
        if (data == "ACM_DBLP"):
            S1_feat = train_features["ACM"][0]
            S2_feat = train_features["DBLP"][0]
        elif (data == "Douban Online_Offline"):
            S1_feat = train_features["Online"][0]
            S2_feat = train_features["Offline"][0]

        S1_emb = GAE(S1_feat.to(device), adj_norm_1).detach()
        S2_emb = GAE(S2_feat.to(device), adj_norm_2).detach()

        D = torch.cdist(S1_emb, S2_emb, 2)
        if (data == "ACM_DBLP"):
            test_idx = test_pairs[:, 0].astype(np.int32)
            labels = test_pairs[:, 1].astype(np.int32)
        elif (data == "Douban Online_Offline"):
            test_idx = test_pairs[0, :].astype(np.int32)
            labels = test_pairs[1, :].astype(np.int32)
        hitAtOne = 0
        hitAtFive = 0
        hitAtTen = 0
        hitAtFifty = 0
        hitAtHundred = 0
        for i in range(len(test_idx)):
            dist_list = D[test_idx[i]]
            sorted_neighbors = torch.argsort(dist_list).cpu()
            label = labels[i]
            for j in range(100):
                if (sorted_neighbors[j].item() == label):
                    if (j == 0):
                        hitAtOne += 1
                        hitAtFive += 1
                        hitAtTen += 1
                        hitAtFifty += 1
                        hitAtHundred += 1
                        break
                    elif (j <= 4):
                        hitAtFive += 1
                        hitAtTen += 1
                        hitAtFifty += 1
                        hitAtHundred += 1
                        break
                    elif (j <= 9):
                        hitAtTen += 1
                        hitAtFifty += 1
                        hitAtHundred += 1
                        break
                    elif (j <= 49):
                        hitAtFifty += 1
                        hitAtHundred += 1
                        break
                    elif (j <= 100):
                        hitAtHundred += 1
                        break
        cur_hitAtOne = hitAtOne / len(test_idx)
        cur_hitAtFive = hitAtFive / len(test_idx)
        cur_hitAtTen = hitAtTen / len(test_idx)
        cur_hitAtFifty = hitAtFifty / len(test_idx)

        if(cur_hitAtOne > best_hitAtOne): best_hitAtOne = cur_hitAtOne
        if (cur_hitAtFive > best_hitAtFive): best_hitAtFive = cur_hitAtFive
        if (cur_hitAtTen > best_hitAtTen): best_hitAtTen = cur_hitAtTen
        if (cur_hitAtFifty > best_hitAtFifty): best_hitAtFifty = cur_hitAtFifty
    print("The best results achieved:")
    print("Hit@1: ", end="")
    print(best_hitAtOne)
    print("Hit@5: ", end="")
    print(best_hitAtFive)
    print("Hit@10: ", end="")
    print(best_hitAtTen)
    print("Hit@50: ", end="")
    print(best_hitAtFifty)


def main(args):
    data = args.dataset
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    train_features = {}
    if (data == "ACM_DBLP"):
        train_set = ["ACM", "DBLP"]
        input_dim = 17
        b = np.load('data/ACM-DBLP.npz')
        train_features["ACM"] = [torch.from_numpy(b["x1"]).float()]
        train_features["DBLP"] = [torch.from_numpy(b["x2"]).float()]
        test_pairs = b['test_pairs'].astype(np.int32)
        NUM_HIDDEN_LAYERS = 12
        HIDDEN_DIM = [1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024]
        output_feature_size = 1024
        lr = 0.0001
        epoch = 50
    elif (data == "Douban Online_Offline"):
        a1, f1, a2, f2, test_pairs = load_douban()
        f1 = f1.A
        f2 = f2.A
        train_set = ["Online", "Offline"]
        input_dim = 538
        test_pairs = torch.tensor(np.array(test_pairs, dtype=int)) - 1
        test_pairs = test_pairs.numpy()
        train_features["Online"] = [torch.from_numpy(f1).float()]
        train_features["Offline"] = [torch.from_numpy(f2).float()]
        NUM_HIDDEN_LAYERS = 6
        HIDDEN_DIM = [512,512,512,512,512,512,512]
        output_feature_size = 512
        lr = 0.0001
        epoch = 20
    print("Loading training datasets")
    train_loader = {}
    for dataset in train_set:
        train_loader[dataset] = [load_adj(dataset)]
    model = TGAE(NUM_HIDDEN_LAYERS,
                input_dim,
                HIDDEN_DIM,
                output_feature_size).to(device)
    print("Generating training features")
    print("Fitting model")
    fit_TGAE_subgraph(data, len(train_set) * (1 + 1), model, epoch, train_loader, train_features, device,
            lr,test_pairs)

def parse_args():
    parser = argparse.ArgumentParser(description="Run T-GAE for subgraph matching task")
    parser.add_argument('--dataset', type=str, default="ACM_DBLP", help='Choose from {ACM_DBLP, Douban Online_Offline}')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)