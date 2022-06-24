# Image Retrieval
The image retrieval pipeline consists of several distinct steps:

## ResNet50 inference to generate database and query set arrays
We utilize a native [PyTorch inference](R50_inference.py) to generate the database and query sets for image retrieval. An example for performing 
retrieval on ImageNet-4K with an MRL pretrained model is provided below:

```
python R50_inference.py --model_path=path-to-model/ --dataset_name=imagenet4k --mrl=1 --efficient=0 
```
This will generate train (database) and val (query) arrays for the vector representations and labels for ImageNet-4K, ie
`imagenet1k_val_mrl1_e0_ff2048-X.npy`, `imagenet1k_val_mrl1_e0_ff2048-y.npy`, `imagenet1k_train_mrl1_e0_ff2048-X.npy`, and `imagenet1k_train_mrl1_e0_ff2048-y.npy`

## Database Index and Search
The arrays generated above are used to create an index file of the database, as shown in `faiss_nn.ipynb`. FAISS requires an 
index type (Exact L2, HNSW32) and a shortlist length $k$ to search the indexed database for each query in the query set. Note that we utilize GPU search 
for exact search, which is currently unsupported for HNSW. The $k$-length shortlist for all desired representation sizes is saved to disk to be used 
for downstream adaptive retrieval or metric computation as, for example, `neighbors/exactl2_16dim-2048-NN_imagenet1k.csv`.

## Adaptive Retrieval
In an attempt to achieve equivalent performance to shortlisting at higher representation sizes with reduced MFLOPs, we perform adaptive retrieval by, for example, retrieving a shortlist $k = 200$ with rep. size $D_r = 16$ followed by reranking with a higher capacity representation $D_s = 2048$. The reranked k-NN shortlist is saved to disk as `8dim-reranked2048_200shortlist_imagenetv2_exactl2.csv` as shown in `reranking.ipynb`.

In an attempt to remove supervision in choosing $D_r$ and $D_s$, we utilize **Funnel Retrieval.** Funnel retrieval thins out the initial shortlist by a 
repeated re-ranking and shortlisting with a series of increasing capacity representations. For example, retrieval with $D_r = 8$ followed by a funnel with Rerank Cascade *= [16, 32, 64, 128, 2048]* and Shortlist Cascade *= [200, 100, 50, 25, 10]* would be saved as  
`8dim-cascade[16,32,64,128,2048]_shortlist[200,100,50,25,10]_imagenet1k_exactl2.csv`

## Metric Computation
We compute mAP@k, precision@k, recall@k, and top-k accuracy of the k-NN shortlist for various values of $k$, as shown in
`compute_metrics.ipynb`, with flags for model configuration, dataset, index type, and retrieval configuration required. The code loads database labels
`imagenet1k_train_mrl1_e0_ff2048-y.npy` and query labels `imagenet1k_val_mrl1_e0_ff2048-y.npy` alongside the k-NN shortlist generated via FAISS retrieval 
(`neighbors/exactl2_16dim-2048-NN_imagenet1k.csv` as in the example above) or 
after reranking in the steps above.
