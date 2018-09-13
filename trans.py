import SimpleITK as sitk
import numpy as np
import cv2
import csv
import os

with open('xxxx.csv','r') as csvfile:             #read csv file containing patients' info
    reader = csv.reader(csvfile)
    columns = []
    labels = []
    for row in reader:
        columns.append(row[1])
        labels.append(row[2])

columns = columns[1:]

def convert_from_dicom_to_jpg(img,low_window,high_window,save_path):
    lungwin = np.array([low_window*1.,high_window*1.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])    #Normalized
    newimg = (newimg*255).astype('uint8')                #Extend the pixel value to[0,255]
    cv2.imwrite(save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

if __name__ == '__main__':
    path = '.../ROI/'                  #address
    filenames = os.listdir(path)
    for filename in filenames:
        #print(filename)
        
        #The following is to convert the corresponding dicom format image into jpg
        #Read dicom file
        dcm_image_path =  path + filename
        portion = os.path.splitext(filename)  #Separate file names and suffixes
        #print(portion)
        
        if portion[1] ==".dcm":
            newname = portion[0]+".jpg"
            #print(newname)
            output_jpg_path = '.../jpg/'+newname
            ds_array = sitk.ReadImage(dcm_image_path)         #Read information about dicom files
            img_array = sitk.GetArrayFromImage(ds_array)      #Get array
            shape = img_array.shape
            img_array = np.reshape(img_array, (shape[1], shape[2]))  #Get the height and width in the array
            high = np.max(img_array)
            low = np.min(img_array)
            convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)   #Call the function, convert it to a jpg file and save it to the corresponding path
            #print('FINISHED')

