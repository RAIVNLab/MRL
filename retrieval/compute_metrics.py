import pandas as pd
import numpy as np
import time

config = 'multihead/'
nesting = 1

#nesting_list = [128, 256, 512, 1024, 2048]
nesting_list = [2048]
#nesting_list = [12, 24, 48, 96, 192, 384, 768, 1536]
ret_dim =16
root_dir = "/mnt/disks/retrieval/corrected_fwd_pass/"+config
#root_dir = "/mnt/disks/retrieval/imagenet_knn_classifier/"+config
#root_dir = '/mnt/disks/imagenet4m/corrected_fwd_pass/' + config
#root_dir = '/mnt/disks/retrieval/slimmable_nn_fwd_pass/'
dataset = 'imagenet1k'
index = 'exactl2'
num_classes = 5202

# Block for nested models
if nesting:
    # Database
    # 1.2M x 1 for imagenet1k
    db_labels = np.load(root_dir + dataset + "_train_nesting1_sh0_ff2048-y.npy")
    #db_labels = np.load(root_dir + "slimmable_train/y_train_slimmable.npy")
    print("DB labels: ", db_labels.shape)

    # Query set
    # 50K x 1 for imagenet1k
    query_labels = np.load(root_dir + dataset + "_val_nesting1_sh0_ff2048-y.npy")
    #query_labels = np.load(root_dir + "imagenetv2/imagenetv2_val_nesting1_sh0_ff2048-y.npy")
    #query_labels = np.load(root_dir + "slimmable_val/Y_val_V1.npy")
    print("Query labels: ", query_labels.shape)

# TODO: copied from
def compute_mAP_recall_at_k(val_classes, db_classes, neighbors, k):
    """
    Computes the MAP@k (default value of k=R) on neighbors with val set by seeing if nearest neighbor
    is in the same class as the class of the val code. Let m be size of val set, and n in train.

      val:          (m x d) All the truncated vector representations of images in val set
      val_classes:  (m x 1) class index values for each vector in the val set
      db_classes:   (n x 1) class index values for each vector in the train set
      neighbors:    (k x m) indices in train set of top k neighbors for each vector in val set
    """

    #print("shape of val_classes is:", val_classes.shape)
    #print("shape of db_classes is:", db_classes.shape)
    #print("shape of neighbors is:", neighbors.shape)

    """
    Imagenet:
    shape of val is: (50000, dim)
    shape of val_classes is: (50000, 1)
    shape of db_classes is: (1281167, 1)
    shape of neighbors is: (50000, 100))
    """

    APs = list()
    precision, recall, topk = [], [], []
    for i in range(val_classes.shape[0]): # Compute precision for each vector's list of k-nn
        target = val_classes[i]
        indices = neighbors[i, :][:k]    # k neighbor list for ith val vector
        labels = db_classes[indices]
        matches = (labels == target)

        # map upper bound
#         hits = np.sum(matches)
#         ideal_matches = np.concatenate([np.ones(hits, dtype=int), np.zeros(k-hits, dtype=int)])
#         tps_ideal = np.cumsum(ideal_matches)
    
        # topk
        hits = np.sum(matches)
        if hits>0:
            topk.append(1)
        else:
            topk.append(0)
            
        # true positive counts
        tps = np.cumsum(matches)
        
        # recall
        recall.append(np.sum(matches)/1300)
        precision.append(np.sum(matches)/k)
        # precision values
        precs = tps.astype(float) / np.arange(1, k + 1, 1)
        APs.append(np.sum(precs[matches.squeeze()]) / k)

    return np.mean(APs), np.mean(precision), np.mean(recall), np.mean(topk)

for dim in nesting_list:
    start = time.time()
    # Block for fixed feature models
    if not nesting:
        db_labels = np.load(root_dir+"imagenet1k_train_nesting0_sh0_ff"+str(dim)+"-y.npy")
        #print("DB labels: ", db_labels.shape)

        query_labels = np.load(root_dir+dataset+"_val_nesting0_sh0_ff"+str(dim)+"-y.npy")
        #print("Query labels: ", query_labels.shape)

    # Load neighbor array and compute top1
    #TODO: rewrite with flags for different datasets
    #neighbors = pd.read_csv(root_dir+"neighbors/"+str(dim)+"dim-2048-NN.csv", header=None).to_numpy()
    #neighbors = pd.read_csv(root_dir+"neighbors/"+str(dim)+"dim-"+str(rerank_dim)+"-NN-hnsw_32.csv", header=None).to_numpy()
    neighbors = pd.read_csv(root_dir+"neighbors/reranked/"+str(ret_dim)+"dim-reranked"+str(dim)+"_2048shortlist_"+ dataset+"_"+index+".csv", header=None).to_numpy()
    #neighbors = pd.read_csv(root_dir+"neighbors/"+index+"_"+str(dim)+"dim-2048-NN_"+dataset+".csv", header=None).to_numpy()

    # Funnel Retrieval
    cascade_nn='8dim-cascade[16, 32, 64, 128, 2048]_[800, 400, 200, 50, 10]shortlist_imagenet4m_exactl2.csv'
    #neighbors = pd.read_csv(root_dir+"neighbors/cascade_naive_policy/"+cascade_nn,header=None).to_numpy()
    print("\nOriginal Dims: ", dim)
    print("Ret Dims: ", ret_dim)

    top1 = db_labels[neighbors[:, 0]]
    print("Top1= ", np.sum(top1 == query_labels) / query_labels.shape[0])
    #print(np.sum(top1 == query_labels) / query_labels.shape[0])
    shortlist = [10, 25, 50, 100]
    for k in shortlist:
        mAP, precision, recall, topk = compute_mAP_recall_at_k(query_labels, db_labels, neighbors, k)
        print("mAP@%d= %f"%(k, mAP))
        print("precision@%d= %f"%(k, precision))
        print("recall@%d= %f"%(k, recall))
        print("top%d= %f"%(k, topk))
    
    #print("Top5 Hits= ", np.sum(top5.any() == query_labels)/ query_labels.shape[0])
    end = time.time()
    print("Eval time for %d= %f" %(dim, (end - start)))

