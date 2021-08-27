from gene2gauss.model_mv import Gene2Gauss_mv
from gene2gauss.utils import *
from gene2gauss.config import args
from gene2gauss.minibatch import MiniBatchIterator
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import optimizers
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

# in this version training is done on highly correlated edges and
# validation is done on external datasets (TM or HumanNet)
if __name__ == '__main__':

    print(args)
    # loading input networks
    adj_mats=[]
    adj_2hops = []
    adj_mats = load_adjacencies(args.inpath)
    two_hop = False

    adj_dtype = adj_mats[0].dtype
    num_nodes = adj_mats[0].shape[0]
    val_edges, val_flag = [], False
    hidden_dims = args.hid_dims

    # generating identity features
    if args.feat_file == None:
        X = identity_features = np.identity(adj_mats[0].shape[0])
    else:
        X = np.load(args.feat_file, allow_pickle=True)
    print("*"*3, "Number of features:", X.shape, "*"*3)

    if args.sparse:
        X = sp.csr_matrix(X)

    train_edges, train_labels, val_edges, val_labels, test_edges, test_labels = split_adjacencies(adj_mats, threshold=args.threshold)
    train_edges = np.concatenate((train_edges, val_edges))
    train_labels = np.concatenate((train_labels, val_labels))
    val_edges = test_edges
    val_labels = test_labels
    print("\tTraining data size:", train_edges.shape, "Validation data size:", val_edges.shape)
    print("\t", Counter(train_labels), Counter(val_labels))

    train_edges_ext, val_edges_ext, test_edges_ext, train_labels_ext, val_labels_ext, test_labels_ext = load_test_edges(args.val_file)
    test_edges = np.concatenate((train_edges_ext, val_edges_ext, test_edges_ext))
    test_labels = np.concatenate((train_labels_ext, val_labels_ext, test_labels_ext))
    print("\tTesting data size:", test_edges.shape)
    print("\t",Counter(test_labels))

    pos_train_edges = train_edges[train_labels==1]
    neg_train_edges = train_edges[train_labels==0]
    # initiating the MiniBatch iterator
    batch_iterator = MiniBatchIterator(adj_mats, train_edges=[pos_train_edges, neg_train_edges], pre_flag=True)

    if args.enc in ['multiGCN', 'multiGAT']:
        two_hop = True
        print("*"*5, "Generating 2-hop neighborhoods", "*"*5)
        adj_2hops = [get_k_hops(adj, 2, keep_weighted=False) for adj in adj_mats]
        if args.enc == 'multiGCN':
            adj_2hops = preprocess_gcn(adj_2hops, sparse=args.sparse)
        else:
            adj_2hops = preprocess_gat(adj_2hops, num_nodes, args.sparse)

    if args.enc in ['GCN', 'multiGCN']:
        adj_mats = preprocess_gcn(adj_mats, args.sparse)
    else:
        adj_mats = preprocess_gat(adj_mats, num_nodes, args.sparse)

    print("*"*10, "Before loading the model", "*"*10)
    get_memory_info()

    # initiating gene2gauss model
    gene2gauss = Gene2Gauss_mv(adj_mats, adj_2hops, X, L=args.emb_size, n_hidden=hidden_dims,
                               dropout_hid=args.dropout_hid, dropout_enc=args.dropout_enc,
                               enc=args.enc, reduce=args.reduce, sparse_flag=args.sparse,
                               two_hop=two_hop, verbose=True, batch_flag=True)

    optimizer = optimizers.Adam(lr=1e-2)

    print("*" * 10, "After loading the model", "*" * 10)
    get_memory_info()

    reduce = args.reduce if args.reduce is not None else 'NA'
    outfile_mu = os.path.join(args.outpath, "dim_" + str(args.emb_size) + "_enc_" + args.enc +
                              "_hid_" + "_".join([str(dim) for dim in hidden_dims]) + "_drop_" + str(args.dropout_hid) + "_reduce_" + reduce + "_mu.npy")
    outfile_sigma = os.path.join(args.outpath, "dim_" + str(args.emb_size) + "_enc_" + args.enc +
                                 "_hid_" + "_".join([str(dim) for dim in hidden_dims]) + "_drop_" + str(args.dropout_hid) + "_reduce_" + reduce + "_sigma.npy")
    metrics_file = os.path.join(args.outpath, "dim_" + str(args.emb_size) + "_enc_" + args.enc +
                                "_hid_" + "_".join([str(dim) for dim in hidden_dims]) + "_drop_" + str(args.dropout_hid) + "_reduce_" + reduce + "_metrics.txt")
    plt_file = os.path.join(args.outpath, "dim_" + str(args.emb_size) + "_enc_" + args.enc +
                            "_hid_" + "_".join([str(dim) for dim in hidden_dims]) + "_drop_" + str(args.dropout_hid) + "_reduce_" + reduce + "_lossplt.png")

    early_stopping_score_max = -float('inf')
    tolerance = args.tolerance
    batch_size = args.batch_size
    num_batches = round(float(num_nodes/batch_size))
    #num_batches=1
    print("Number of batches:", num_batches)

    epochs = args.max_iter
    batch_num = 0

    metrics = []
    for epoch in range(epochs):
        avg_loss = 0.0
        pbar = tqdm(range(num_batches))
        for i in pbar:
            batch_nodes, pos_edges, neg_edges = batch_iterator.next_batch()
            with tf.GradientTape() as tape:
                loss = gene2gauss((batch_nodes, pos_edges, neg_edges, True))
            # updating gradients
            grads = tape.gradient(loss, gene2gauss.trainable_variables)
            optimizer.apply_gradients(zip(grads, gene2gauss.trainable_variables))
            avg_loss += loss
            pbar.set_description("Batch: %d" % (i+1))

        avg_loss = avg_loss/num_batches
        roc_auc, precision = gene2gauss.link_prediction(val_edges, val_labels)
        tf.print("\t****** Epoch: %d, Loss: %.3f, Precision: %.3f, AUROC: %.3f *****"
                 %((epoch+1), avg_loss, precision, roc_auc))
        metrics.append([(epoch+1), round(float(avg_loss),3), precision, roc_auc])

        # implementing early stopping
        early_stopping_score = roc_auc + precision

        # resetting early stopping flags if higher AUC and/or AP values are achieved in the current epoch
        if early_stopping_score > early_stopping_score_max:
            early_stopping_score_max = early_stopping_score
            tolerance = args.tolerance
        else:
            tolerance -= 1

        if tolerance == 0:
            print("*" * 10, "Early Stopping...", "*" * 10)
            break

    print("*" * 10, " Saving Embeddings ", "*" * 10)
    np.save(outfile_mu, gene2gauss.mu)
    np.save(outfile_sigma, gene2gauss.sigma)

    print("*" * 10, " Saving Metrics ", "*" * 10)
    metrics_df = pd.DataFrame(metrics, columns = ["Epoch", "Loss", "Precision", "ROC-AUC"])
    metrics_df.to_csv(metrics_file, sep="\t", index=False)

    test_roc_auc, test_precision = gene2gauss.link_prediction(test_edges, test_labels)
    print("*"*10, "Test Metrics", "*"*10)
    print("AUROC: %.3f, Precision: %.3f " %(test_roc_auc, test_precision))

    metrics_df.plot(x='Epoch', y="Loss", linestyle="-", marker='o', markerfacecolor='black')
    plt.savefig(plt_file)

# python training_unsupervised.py --inpath /home/aniljegga2/bigdataserver/Sudhir_stuff/networks_gene2gauss/soft_threshold_networks/
# --feat_file /home/aniljegga2/bigdataserver/Sudhir_stuff/networks_gene2gauss/features_krasnow1.npy --sparse
# --val_file /home/aniljegga2/bigdataserver/Sudhir_stuff/networks_gene2gauss/validation_edges.npz --hid_dims 128
# --outpath /home/aniljegga2/bigdataserver/Sudhir_stuff/g2g_results_self_super/gene2gauss_results/gcn