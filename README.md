# TeamXX
- Haolin Jiang (st176083)
- Diandian Guo (st175733)

# Requirements

Our code is based on Pytorch. You can install pytorch with "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116".

# Dataset

Our simple model diabetic retinopathy experiments uses the lab dataset, you can download the dataset from this link: https://ilias3.uni-stuttgart.de/ilias.php?ref_id=3124763&page=Datasets&wpg_id=10386&cmd=downloadFile&cmdClass=ilwikipagegui&cmdNode=192:tv:195&baseClass=ilwikihandlergui&file_id=il__file_2896881 

Please do the preprocessing before feeding the data into our network. We have two preprocessing steps in the diabetic_retinopathy/input_pipeline. Step 1: remove the black edges of the images(preprocessing.py). Step2: calculate the mean and variance of the images(find_mean_and_std.py). 

To run the HAPT experiments, please dowbnload the HAPT dataset from: https://ilias3.uni-stuttgart.de/ilias.php?ref_id=3124763&page=Datasets&wpg_id=10386&cmd=downloadFile&cmdClass=ilwikipagegui&cmdNode=192:tv:195&baseClass=ilwikihandlergui&file_id=il__file_2897015

Our mean teacher expriments uses additional Kaggle dataset for unsupervised training and testing. We provide preprocessed Kaggle dataset here: 

728x728 size:  https://drive.google.com/drive/folders/1smBxnf8LYucqHuDMT5I_Z5G5FDjLW5AF?usp=sharing

512x512 size: https://drive.google.com/drive/folders/1rnmoMBTaO8i2J4EvBsGk_HEQmMC8dRZm?usp=sharing

Please contact Haolin for permission before downloading the dataset from his google drive.



# How to run

Put the downloaded datasets in ./Labdata.

For the simple model diabetic retinopathy expriment, run simple_diabetic branch/diabetic_retinopathy/train.py

For the mean teacher experiment, run master/diabetic_retinopathy/mean_teacher.py

For the HAPT experiment, run  master/HAPT/train.py

# checkpoints

For ensemble learning, please place the saved checkpoints in the folder and define the folder path in dl-lab-22w-team15/diabetic_retinopathy/evaluation/ensemble.py.


If you need the pretrained model, please ask Haolin for help.



# Results
Simple diabetic retinopathy classification model:                    71% accuracy on labdata test set,   27% accuracy on Kaggle test set                         

Transferred diabetic retinopathy classification model from efficientnet b7:   89% accuracy on labdata test set,   37% accuracy on Kaggle test set

Mean teacher model:              88% accuracy on labdata test set,   80% accuracy on Kaggle test set

HAPT:  96% accuracy on HAPT test set
