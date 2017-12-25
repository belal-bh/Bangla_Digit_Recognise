#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  
#  Helper functions for dataset
#
#  author  : Md. Belal Hossain
#  GitHub  : https://github.com/belal-bh/Bangla_Digit_Recognise
#  Facebook: https://www.facebook.com/belal.bh.pro
#  Date    : December 5, 2017
#  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#========================= Imports ========================
import os,cv2
import numpy as np
from sklearn.utils import shuffle
from keras.utils import np_utils


#==========================================================



#======================= Function for generate dataset =============
def generate_dataset(path, img_shape=(28,28),num_channel=1,num_classes=10):
    data_dir_list = os.listdir(path)
    print('data_dir_list:',data_dir_list)
    
    # list of image data 
    img_data_list=[]
    # list of number of images in each cls folder that means number of image in every class
    num_im = []
    for dataset in data_dir_list:
        img_list=os.listdir(path+'/'+ dataset)
        print('Loaded the images of dataset-'+'{}\n'.format(dataset))
        num_im.append(len(img_list))
        for img in img_list:
            input_img=cv2.imread(path +'/'+ dataset + '/'+ img )
            input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_resize=cv2.resize(input_img,(28,28))
            img_data_list.append(input_img_resize)
            
    #print('img_data_list[:1]:=>', img_data_list[:1])
    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    
    #print(img_data[:1,:1,:])
    #print(img_data.shape)


    if num_channel==1:
        img_data= np.expand_dims(img_data, axis=4) 
        print(img_data.shape)

    # Assigning Labels
    num_of_samples = img_data.shape[0]
    labels = np.ones((num_of_samples,),dtype='int64')
    #print('labels=',labels)
    print('num_im:',num_im)
    # Generate labels automatically
    for i in range(num_classes):
        labels[sum(num_im[:i]):sum(num_im[:i+1])]=i
        print('Class=',i,'Total=',len(labels[sum(num_im[:i]):sum(num_im[:i+1])]))
        #print('labels:\n',labels[sum(num_im[:i]):sum(num_im[:i+1])])

    print('Tolal number of element of labels: ',len(labels))
    
    #Shuffle the dataset
    img_data,labels = shuffle(img_data,labels, random_state=2)
    # convert class labels to on-hot encoding
    labels = np_utils.to_categorical(labels, num_classes)
    
    return img_data,labels

#==================================================================