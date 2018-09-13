
#ROI
import os

#read csv files
import csv
with open('xxxxxx.csv','r') as csvfile:      #csv file of patients' info data
    reader = csv.reader(csvfile)
    columns = []
    labels = []
    for row in reader:
        columns.append(row[1])          #find 
        labels.append(row[2])


def detect_walk(dir_path, target_str):
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            if target_str in filename:
                return root + '/' + filename

#split column
posible_folders = []
columns = columns[1:]
labels = labels[1:]
results = []
for ith in range(len(columns)):
    if columns[ith] == '':
        continue
    head = columns[ith].split('/')
    label = labels[ith]
    id = head[0]
    type = head[-1]
    # dir = r'/home/sarp/Projects/CBIS-DDSM/Data/' + id
    #dir = '/Users/sunying/Desktop/db3/input/' + id
    #dir = '/Users/sunying/Desktop/CBIS-DDSM/' + id
    dir = '/Volumes/TOSHIBA EXT/CBIS-DDSM/' + id

    old_file = detect_walk(dir, '000000.dcm')
    print(old_file)
    if old_file is None:
        continue
    result = '/Volumes/TOSHIBA EXT/ROI/' + label + id + '.dcm'
    print(result)
    order = 'cp "' + old_file + '" "' + result + '"'
    print(order)
    os.system(order)
'''

#FULL IMAGE

import os
#read csv files
import csv
with open('/Users/sunying/Desktop/db3/A.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    columns = []
    labels = []
    for row in reader:
        columns.append(row[12])
        labels.append(row[9])

def detect_walk(dir_path, target_str):
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            if target_str in filename:
                return root + '/' + filename
#split column
posible_folders = []
columns = columns[1:]
labels = labels[1:]
results = []
for ith in range(len(columns)):
    if columns[ith] == '':
        continue
    head = columns[ith].split('/')
    label = labels[ith]
    id = head[0]
    type = head[-1]
    # dir = r'/home/sarp/Projects/CBIS-DDSM/Data/' + id
    #dir = '/Users/sunying/Desktop/db3/input/' + id
    #dir = '/Users/sunying/Desktop/CBIS-DDSM/' + id
    dir = '/Volumes/TOSHIBA EXT/CBIS-DDSM/' + id
    
    old_file = detect_walk(dir, '000000.dcm')
    print(old_file)
    if old_file is None:
        continue
    result = '/Volumes/TOSHIBA EXT/full/' + label + id + '.dcm'
    print(result)
    order = 'cp "' + old_file + '" "' + result + '"'
    print(order)
    os.system(order)
'''
