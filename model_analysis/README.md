# Model Analysis
Here provide jupyter notebooks, which contains performance visualization such as GradCAM images (for checkpoint models), superclass performance, model cascades and oracle upper bound.  

#### [GradCAM.ipynb](GradCAM.ipynb)

This notebook visualizes model attribution for each image. We beforehand store each image as torch tensor, arranged class-wise, and therefore please do so before running this script. Note that the example in the notebook is not the one shown in the paper.

#### [Cascade Performance Paper.ipynb](<./Cascade Performance Paper.ipynb>)
This notebook evaluates our greedy scheme based on maximum probability thresholding for model cascading. Make sure to have softmax predictions stored for given model under consideration. 

#### [Custom SuperClass Performance.ipynb](<./Custom SuperClass Performance.ipynb>)
Based on wordnet heirarchy, we evaluate our MRL model on 30 randomly chosen superclasses. The code is based on [robustness package](https://github.com/MadryLab/robustness). 	  

#### [Oracle Upper Bound Performance.ipynb](<./Oracle Upper Bound Performance.ipynb>)
We compute oracle performance (the maximum possible accuracy) for MRL with ideal routing. 
