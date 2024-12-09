# QCmriQC
Class project using deep learning (CNN &amp; autoencoder) to classify T1w MRI structural images as usable or not. Inspired by Drs. Esteban and Garcia. The goal is to incorporate IQMs from MRIQC proper instead of raw PNG images converted via med2image ([Link](https://github.com/FNNDSC/med2image))
# Implementation
After downloading files (train.py utils.py, models folder) run via bash directly in terminal/cmd prompt. Returns plots and tables on training and validation performance, values of test performance (confusion matrix), prediction values for test set.

<p> $ python /PATH-TO-train.py <br>
  --model 		# simple_cnn or vgg16_untrained or vgg16_default or autoencoder or resnet <br>
 	--data_dir 	# /PATH-TO-PARSED-DATA/ <br>
  --epochs 	# number of epochs to use <br>
  --patience	# number of patience epochs <br>
  --save_dir	# /PATH-TO-WHERE-YOU-WANT-2-SAVE <p/>
    
# Models
These include a simple architecture 2 layer CNN model, VGG-16 (default & pretrained), ResNet-18 (pretrained), as well as an autoencoder model. 
# Data
This data was pulled publically from "The Dual Mechanisms of Cognitive Control" dataset, a theoretically-guided within-subject task fMRI battery by Jo Etzel & Todd Braver
## CITATION
##### Etzel, J.A., Brough, R.E., Freund, M.C. et al. The Dual Mechanisms of Cognitive Control dataset, a theoretically-guided within-subject task fMRI battery. Sci Data 9, 114 (2022). https://doi.org/10.1038/s41597-022-01226-4
([Link](https://www.nature.com/articles/s41597-022-01226-4))

Accepted scans found [here](https://openneuro.org/datasets/ds003465/versions/1.0.6)

Rejected scans found [here](https://osf.io/a7w39)

Original thread found ([here](https://neurostars.org/t/training-sets-for-manual-qc-of-mri-data/22603))


