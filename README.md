# TeamXX
- Haolin Jiang (st176083)
- Diandian Guo (st175733)

# Requirements

Our code is based on Pytorch. You can install pytorch with "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116".

# Dataset

Our simple model diabetic retinopathy experiments uses the lab dataset, you can download the dataset from this link: https://ilias3.uni-stuttgart.de/ilias.php?ref_id=3124763&page=Datasets&wpg_id=10386&cmd=downloadFile&cmdClass=ilwikipagegui&cmdNode=192:tv:195&baseClass=ilwikihandlergui&file_id=il__file_2896881 

Please do the preprocessing before feeding the data into our network. We have two preprocessing steps in the diabetic_retinopathy/input_pipeline. Step 1: remove the black edges of the images(preprocessing.py). Step2: calculate the mean and variance of the images(find_mean_and_std.py).

To run the HAPT experiments, please dowbnload the HAPT dataset from: https://ilias3.uni-stuttgart.de/ilias.php?ref_id=3124763&page=Datasets&wpg_id=10386&cmd=downloadFile&cmdClass=ilwikipagegui&cmdNode=192:tv:195&baseClass=ilwikihandlergui&file_id=il__file_2897015

Our mean teacher expriments uses additional Kaggle dataset for unsupervised training and testing.




# How to run the code
to do

# Results
to do
