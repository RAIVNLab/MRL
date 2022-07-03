# Model Analysis
We provide Jupyter notebooks, which contain performance visualization via [GradCAM](https://github.com/jacobgil/pytorch-grad-cam) images (for checkpoint models), superclass performance, model cascades and oracle upper bound. Please be sure to have predictions and softmax probabilities saved beforehand (using our save flags). In addition, for adaptive classification using cascades, use `--tta` flag while saving the above. 

#### [GradCAM](GradCAM.ipynb)
This notebook visualizes model attribution for each image. As required preprocessing, we store each image as a torch tensor, arranged class-wise. This notebook illustrates that using smaller representation size for classification can result in model confusion between classes within the same superclass (*e.g.* Rock Python vs Boa Constrictor as in Figure 9.b below).

<p align="center">
<img src="../images/gradcam.jpeg" width="1024"/>
</p>

#### [Adaptive Classification with Cascades](<./Cascade_Performance_Paper.ipynb>)
This notebook evaluates our greedy scheme for model cascading, based on maximum probability thresholding. This notebook requires the softmax predictions for the model under consideration. 

<p align="center">
<img src="../images/adaptive_classification.png" width="512"/>
</p>

#### [Custom SuperClass](<./Custom_SuperClass_Performance.ipynb>)
Based on [WordNet](https://www.nltk.org/howto/wordnet.html) hierarchy, we evaluate the MRL model on 30 randomly chosen superclasses. The code is based on the [MadryLab robustness package](https://github.com/MadryLab/robustness). 	  

#### [Oracle Upper Bound](<./Oracle_Upper_Bound_Performance.ipynb>)
We compute oracle performance, *i.e.* the maximum possible achievable accuracy for MRL with ideal routing for appropriate rep. size.

