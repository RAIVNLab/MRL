# Model Analysis
We provide Jupyter notebooks, which contain performance visualization via GradCAM images (for checkpoint models), superclass performance, model cascades and oracle upper bound.  

#### [GradCAM.ipynb](GradCAM.ipynb)
This notebook visualizes model attribution for each image. As required preprocessing, we store each image as a torch tensor, arranged class-wise. Note that the example in the notebook is not the one shown in the paper.

**TODO @ GB** please add a comment why and how it is different from paper

#### [Cascade Performance](<./Cascade_Performance_Paper.ipynb>)
This notebook evaluates our greedy scheme for model cascading, based on maximum probability thresholding. This notebook requires the softmax predictions for the model under consideration. 

#### [Custom SuperClass Performance](<./Custom_SuperClass_Performance.ipynb>)
Based on WordNet hierarchy, we evaluate our MRL model on 30 randomly chosen superclasses. The code is based on this [robustness](https://github.com/MadryLab/robustness) project. 	  

**TODO @ GB:** add WordNet link?

#### [Oracle Upper Bound Performance](<./Oracle_Upper_Bound_Performance.ipynb>)
We compute oracle performance, i.e. the maximum possible achievable accuracy for MRL with ideal routing for appropriate rep. size.
