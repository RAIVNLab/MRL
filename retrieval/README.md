# Image Retrieval
The image retrieval pipeline consists of several distinct steps:

## R50 inference to generate database and query set arrays
We utilize a native [PyTorch inference](R50_inference.py) to generate the database and query sets for image retrieval. An example for performing 
retrieval on ImageNet-4K with an MRL pretrained model is provided below:

```
python R50_inference.py --model_path=path-to-model/ --dataset_name=imagenet4k --mrl=1 --efficient=0 
```
This will generate train and val arrays for the vector representations and labels for ImageNet-4K, ie
`imagenet1k_val_mrl1_e0_ff2048-X.npy`, `imagenet1k_val_mrl1_e0_ff2048-y.npy`, `imagenet1k_train_mrl1_e0_ff2048-X.npy`, and `imagenet1k_train_mrl1_e0_ff2048-y.npy`

## Database index and search
The arrays generated above are used to create an index file of the database, as shown in `faiss_nn.ipynb`. FAISS requires an 
index type (Exact L2, HNSW_32) and a shortlist length $k$ to search the indexed database for each query in the query set. Note that we utilize GPU search 
for exact search, which is currently unsupported for HNSW. The $k$-length shortlist for all desired representation sizes is saved to disk to be used 
for downstream adaptive retrieval or metric computation.

## Adaptive Retrieval
In an attempt to achieve equivalent performance to shortlisting at higher representation sizes with reduced MFLOPs, we perform adaptive retrieval by, for example, retrieving
a shortlist $k$ with rep. size $D_r = 16$ followed by reranking with a higher capacity representation $D_s = 2048$. 

In an attempt to remove supervision in choosing $D_r$ and $D_s$, we utilize **Funnel Retrieval.** Funnel retrieval thins out the initial shortlist by a 
repeated re-ranking and shortlisting with a series of increasing capacity representations.

## Metric Computation
We compute mAP@k, precision@k, recall@k, and top-k accuracy of the k-NN shortlist for various values of $k$. Metric computation can be run simply as
`python compute_metrics.py`, with flags for model configuration, dataset, index type, and retrieval configuration required. The code loads database labels
`imagenet1k_train_mrl1_e0_ff2048-y.npy` and query labels `imagenet1k_val_mrl1_e0_ff2048-y.npy` alongside the k-NN shortlist generated via FAISS retrieval or 
after reranking in the steps above.
