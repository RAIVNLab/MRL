import pandas as pd
import numpy as np
import time

#TODO: delete this file
CONFIG = 'mrl' # one of ['mrl', 'mrl_e', 'ff']
NESTING = CONFIG in ['mrl', 'mrl_e']
ROOT_DIR = "path_to_db_and_query_files/" + CONFIG + "/"
DATASET = 'imagenet1k'
SEARCH_INDEX = 'exactl2' # one of ['exactl2', 'hnsw_8', 'hnsw_32']
EVAL_CONFIG = 'vanilla' # ['vanilla', 'reranking', 'funnel']

'''
nesting_list is used in two ways depending on the config:
1. vanilla: nesting_list = scales at which we retrieve representations for all images
2. reranking: nesting_list = scales at which we rerank representations for all images
3. funnel: unused
'''
if EVAL_CONFIG in ['vanilla', 'reranking']:
    nesting_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
else:
    # funnel retrieval
    CASCADE_NN_FILE = '8dim-cascade[16, 32, 64, 128, 2048]_[800, 400, 200, 50, 10]shortlist_imagenet4m_exactl2.csv'
    nesting_list = [2048] # for funnel, we evaluate a single config at a time

'''
ret_dim is used in two ways depending on the config:
1. vanilla: unused
2. reranking: retrieve representations of size ret_dim and rerank with nesting_list
3. funnel: unused
'''
ret_dim = 16

# Load database and query set for nested models
if NESTING:
    # Database: 1.2M x 1 for imagenet1k
    db_labels = np.load(ROOT_DIR + DATASET + "_train_nesting1_sh0_ff2048-y.npy")
    # Query set: 50K x 1 for imagenet1k
    query_labels = np.load(ROOT_DIR + DATASET + "_val_nesting1_sh0_ff2048-y.npy")

def compute_mAP_recall_at_k(val_classes, db_classes, neighbors, k):
    """
    Computes the MAP@k (default value of k=R) on neighbors with val set by seeing if nearest neighbor
    is in the same class as the class of the val code. Let m be size of val set, and n in train.

      val:          (m x d) All the truncated vector representations of images in val set
      val_classes:  (m x 1) class index values for each vector in the val set
      db_classes:   (n x 1) class index values for each vector in the train set
      neighbors:    (k x m) indices in train set of top k neighbors for each vector in val set
    """

    """
    ImageNet-1K:
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
    # Load database and query set for fixed feature models
    if not NESTING:
        db_labels = np.load(ROOT_DIR + DATASET + "_train_nesting0_sh0_ff" + str(dim) + "-y.npy")
        query_labels = np.load(ROOT_DIR + DATASET + "_val_nesting0_sh0_ff" + str(dim) + "-y.npy")

    # Load neighbors array and compute metrics
    if EVAL_CONFIG == 'reranking':
        print("\nRet Dim: ", ret_dim)
        print("Rerank dim: ", dim)
        neighbors = pd.read_csv(ROOT_DIR + "neighbors/reranked/" + str(ret_dim) + "dim-reranked" + str(dim) + "_2048shortlist_"
                                + DATASET + "_" + SEARCH_INDEX + ".csv", header=None).to_numpy()
    elif EVAL_CONFIG == 'vanilla':
        print("\nRet Dim: ", dim)
        neighbors = pd.read_csv(ROOT_DIR + "neighbors/" + SEARCH_INDEX + "_" + str(dim) + "dim_2048shortlist_"
                                + DATASET + ".csv", header=None).to_numpy()
    else:
        neighbors = pd.read_csv(root_dir +"neighbors/funnel_retrieval/" + CASCADE_NN_FILE, header=None).to_numpy()

    top1 = db_labels[neighbors[:, 0]]
    print("Top1= ", np.sum(top1 == query_labels) / query_labels.shape[0])

    shortlist = [10, 25, 50, 100] # compute metrics at different shortlist lengths
    for k in shortlist:
        mAP, precision, recall, topk = compute_mAP_recall_at_k(query_labels, db_labels, neighbors, k)
        print("mAP@%d = %f"%(k, mAP))
        print("precision@%d = %f"%(k, precision))
        print("recall@%d = %f"%(k, recall))
        print("top%d = %f"%(k, topk))

    end = time.time()
    print("Eval time for %d= %f" %(dim, (end - start)))

