## Evaluating pre-trained checkpoints
We also provide path to the saved checkpoints which were used to generate the results in our paper. However, the MRL Linear layer had a slightly different re-factored [code](eval_ckpts/NestingLayer.py), therefore we have separate scripts to directly evaluate them. Saved model ckpts can be found [here](https://drive.google.com/drive/folders/1IEfJk4xp-sPEKvKn6eKAUzvoRV8ho2vq?usp=sharing). 

Here in addition to the scripts, we also provide jupyter notebooks, which contains performance visualization such as GradCAM images (for checkpoint models), superclass performance, model cascades and oracle upper bound.  

### Evaluating MRL models 
```
python eval_MRL.py --path [path to weight checkpoint, need to be .pt file] --dataset [V2/A/Sketch/R/V1] (--efficient)
```

### Jupyter Notebooks 

#### [GradCAM.ipynb](GradCAM.ipynb)

This notebook visualizes model attribution for each image. We beforehand store each image as torch tensor, arranged class-wise, and therefore please do so before running this script. 

#### [Cascade Performance Paper.ipynb](<./Cascade Performance Paper.ipynb>)
This notebook evaluates our greedy scheme based on maximum probability thresholding for model cascading. Make sure to have softmax predictions stored for given model under consideration. 

#### [Custom SuperClass Performance.ipynb](<./Custom SuperClass Performance.ipynb>)
Based on wordnet heirarchy, we evaluate our MRL model on 30 randomly chosen superclasses. The code is based on [robustness package](https://github.com/MadryLab/robustness). 	  

#### [Oracle Upper Bound Performance.ipynb](<./Oracle Upper Bound Performance.ipynb>)
We compute oracle performance (the maximum possible accuracy) for MRL with ideal routing. 
