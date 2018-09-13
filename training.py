
#coding=utf-8

import os
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###roc,auc
from sklearn import cross_validation
from scipy import interp
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

data_dir = "ROI/train"   #ROI training
#data_dir = "ROI/test"   #ROI testing
#data_dir = "full/train"  #fullimage train
#data_dir = "full/test"  #fullimage test


train = True   #start training
#train = False   #start testing

model_path = "model/image_model"  #model path



def read_data(data_dir):     #Read images and tags from a folder into a numpy array
    datas = []
    labels = []
    fpaths = []
    num_samples = len(os.listdir(data_dir))
    
    #print(data_dir)
    #print(num_samples)
    datas = np.zeros((num_samples,113,121,1))
    labels = np.zeros((num_samples,1))     # Label information in the file name, for example, 1+xxxx.jpg indicates that the label of the image is 1
    t = 0

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        image = image.resize((121,113))
        data = np.array(image) / 255.0
        #print(data)
        data = np.expand_dims(data, axis=3)
        #print(data)
        #label = int(fname.split("_")[0])  #splitting labels
        label = int(fname.split("+")[0])
        #patient = int(fname.split("_")[2])
     
        datas[t]=data
        labels[t,0]=label
        t = t + 1


    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels

print(data_dir)

fpaths, datas, labels = read_data(data_dir)


# Calculate how many types of pictures
num_classes = len(labels)

# Define Placeholder to store input and tags
datas_placeholder = tf.placeholder(tf.float32, [None, 113, 121, 1])
#print(datas_placeholder)

labels_placeholder = tf.placeholder(tf.float32, [None, 1])

# Container for storing DropOut parameters, 0.25 for training and 0 for testing
dropout_placeholdr = tf.placeholder(tf.float32)

# Convolutional layer with 30 convolution kernels, convolution kernel size 5, activated with Relu
conv0 = tf.layers.conv2d(datas_placeholder, 30, 5, activation=tf.nn.relu)

# max-pooling, the pooling window is 2x2 and the step size is 2x2
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

#Convolutional layer with 30 convolution kernels, convolution kernel size 4, activated with Relu
conv1 = tf.layers.conv2d(pool0, 30, 4, activation=tf.nn.relu)

# max-pooling, the pooling window is 2x2 and the step size is 2x2
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

#Convolutional layer with 30 convolution kernels, convolution kernel size 2, activated with Relu
conv2 = tf.layers.conv2d(pool1, 30, 2, activation=tf.nn.relu)

# max-pooling
pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2])

# Convert 3D features to 1D vectors
flatten = tf.layers.flatten(pool2)
#flatten = tf.layers.flatten(pool1)

# Full connection layer, converted to a feature vector of length 100
fc = tf.layers.dense(flatten, 20, activation=tf.nn.relu)

# Add DropOut to prevent overfitting
dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

# Unactivated output layer
logits = tf.layers.dense(dropout_fc, 1)
#predicted_labels = tf.arg_max(logits, 1)

losses = tf.pow(tf.subtract(logits, labels_placeholder),2)

#print(labels_placeholder.shape, logits.shape)
#while True:
#pass
#losses = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels_placeholder, num_classes),logits=logits)

# average loss
mean_loss = tf.reduce_mean(losses)

# Define the optimizer and specify the loss function to optimize
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(mean_loss)

# for saving and loading models
saver = tf.train.Saver()

with tf.Session() as sess:

    if train:
        print("TRAINING:")
        #Training, initialization parameters
        sess.run(tf.global_variables_initializer())
        #print(datas[0], '----\n',datas[1])
        #print(labels)
        num_samples = len(datas)
        idx = np.zeros((num_samples))
        for i in range(num_samples):
            idx[i] = int(i)
        #feed_datas = np.zeros((num_samples,113,121,1))
        feed_datas = np.zeros((num_samples,200,200,1))
        feed_labels = np.zeros((num_samples,1))
      
     
        for step in range(150):
            #print(optimizer)
            # sample
            train_feed_dict = {
                datas_placeholder: datas,
                labels_placeholder: labels,
                dropout_placeholdr: 0.25
            }
            np.random.shuffle(idx)
            for i in range(num_samples):
                feed_datas[i]=datas[int(idx[i])]
                feed_labels[i]=labels[int(idx[i])]
            #print(idx)
            
            _,mean_loss_val,test = sess.run([optimizer, mean_loss, logits], feed_dict=train_feed_dict)
            print("step = {}\tmean loss = {}".format(step, mean_loss_val))

            if step % 50 == 0:
                saver.save(sess, model_path)
        print("TRAINING'S OVER, MODEL SAVED TO {}".format(model_path))



    else:
        print("TESTING:")
        saver.restore(sess, model_path)
        print("{} LOADED MODEL".format(model_path))
        label_name_dict = {
            0: "malignant",
            1: "benign"
        }

        test_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0
        }

        predicted_logits = sess.run(logits, feed_dict=test_feed_dict)
        count = 0
        y_test = []
        preds = []
        #True label and model prediction label
        for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_logits):
            #real_label_name = label_name_dict[real_label]
            if predicted_label>0.5:
                predicted_label_name = 1
            else:
                predicted_label_name = 0
            #print("real label: %d, predict label: %d" % (real_label[0],predicted_label_name))
                #print("predict label: %d",predicted_label_name)
            #predicted_label_name = label_name_dict[predicted_label]
            if real_label[0] < 0.5:
                real_output = "malignant"
            else:
                real_output = "benign"
            if predicted_label_name < 0.5:
                predict_output = "malignant"
            else:
                predict_output = "benign"
            #print("{}\t{} => {}".format(fpath, real_output, predict_output))
            print("real label: ",real_output,"  predict label: " ,predict_output)
            patient = fpath.split("_")[2:5]
            if real_output != predict_output:
                count = count + 1
                print("wrong prediction of patient: ",patient)
            y_test.append(predicted_label_name)
            preds.append(int(real_label[0]))

        cnf_matrix = confusion_matrix(np.array(y_test), np.array(preds))
        print("confusion_matrix is :")
        print(cnf_matrix)

        #F-measure
        cnf_a = cnf_matrix[0][0]
        cnf_b = cnf_matrix[0][1]
        cnf_c = cnf_matrix[1][0]
        cnf_d = cnf_matrix[1][1]
        P = cnf_a/(cnf_a+cnf_b)
        R = cnf_a/(cnf_a+cnf_c)
        F1 = 2 * P * R / (P + R)
        print("F-measure = ", F1)

        #ROC AUC
        false_positive_rate,true_positive_rate,thresholds=roc_curve(np.array(y_test), np.array(preds))
        roc_auc=auc(false_positive_rate, true_positive_rate)
        plt.title('ROC')
        plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.savefig('xxx.jpg')       #add address of ROC curve
        plt.close(0)


'''
        # Run classifier with cross-validation and plot ROC curves
        
        X = predicted_logits
        n_samples, n_features = X.shape
        random_state = np.random.RandomState(0)
        X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
        
        y = np.array(y_test)
        cv = StratifiedKFold(y, n_folds=5)
        classifier = svm.SVC(kernel='linear', probability=True,
                             random_state=random_state)
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []

        for i, (train, test) in enumerate(cv):
            #train data with SVM model and test
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
 
            # Compute ROC curve and area the curve
            #calculate fpr, tpr, thresold by roc_curve()

            fpr, tpr, thresholds = roc_curve(np.array(y[test]), probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)            #Interpolate the mean_tpr at mean_fpr
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))


        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0                         #the last spot is（1,1）
        mean_auc = auc(mean_fpr, mean_tpr)        #get AUC
        #get ROC curve
        #print(mean_fpr,len(mean_fpr))
        #print(mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()


'''

    


