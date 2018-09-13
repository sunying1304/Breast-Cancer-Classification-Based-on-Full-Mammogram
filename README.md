# Readme

Breast-Cancer-Classification-Based-on-Full-Mammogram

Using Convolutional Neural Networks (CNN) for classification of breast cancer (benign/malignant). 
Most articles in the breast cancer classification were based on ROI (region of interest) that doctors manually generated. This work also help to generate the cluster of ROI automatically.

1.	Dataset
For the first stage, we apply DDSM as training and testing set. DDSM (digital database for screening mammography) database can be obtained online : http://marathon.csee.usf.edu/Mammography/Database.html

2.	File functions
(a)	Content of whole procedure
locate.py: sort out ROI/full mammogram from DDSM dataset
trans.py: convert dicom files into jpeg, can be connected to locate.py by editing “path”
training.py: training, testing, cross-validation, can be connected to trans.py by adding the path of converted images
mask of the breast area and auto ROI.ipynb: providing the mask of breast area and cut out embedded patients’ information as well as generating cluster area by FCM

(b)	Other attempts
FCM clustering.ipynb: fuzzy c-means algorithm
ROI selection based on grey value.ipynb: generate ROI area based on grey value, yet its result is not as satisfying as FCM
converting jpg to csv.ipynb: providing a method to categorize image data into csv file, which will be easier to read
extracting features from ROI or full image (for plotting) .ipynb: visualize selected features
image segmentation – threshold and histogram.ipynb: providing the mask of breast and the histogram image

3.	Methods


4.	Training 
(a)	You will have to install Tensorflow
(b)	Both training and testing is in “training.py” file. Change the value of “training” inside the file if you want to apply different functions.

5.	Examples
Example output of FCM is shown in mask of the breast area and auto ROI.ipynb file.
Example of cross-validation result of auto-generated ROI:


contact: sunying1304@126.com

