# Model Analysis
We provide Jupyter notebooks, which contain performance visualization via GradCAM images (for checkpoint models), superclass performance, model cascades and oracle upper bound.  

#### [GradCAM.ipynb](GradCAM.ipynb)
This notebook visualizes model attribution for each image. We beforehand store each image as torch tensor, arranged class-wise, and therefore please do so before running this script. This notebook illustrates that smaller representation size based classification can get confused within the classes in same superclass (for example, Rock Python vs Boa Constrictor).

#### [Cascade Performance Paper](<./Cascade_Performance_Paper.ipynb>)
This notebook evaluates our greedy scheme based on maximum probability thresholding for model cascading. Make sure to have softmax predictions stored for given model under consideration. 

#### [Custom SuperClass Performance](<./Custom_SuperClass_Performance.ipynb>)
Based on [wordnet](https://www.nltk.org/howto/wordnet.html) heirarchy, we evaluate our MRL model on 30 randomly chosen superclasses. The code is based on [robustness package](https://github.com/MadryLab/robustness). 	  

#### [Oracle Upper Bound Performance](<./Oracle_Upper_Bound_Performance.ipynb>)
We compute oracle performance (the maximum possible accuracy) for MRL with ideal routing. 

