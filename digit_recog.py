#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  
#  Bangla Digit Recognise using Tensorflow Keras API
#  
#  author  : Md. Belal Hossain
#  GitHub  : https://github.com/belal-bh/Bangla_Digit_Recognise
#  Facebook: https://www.facebook.com/belal.bh.pro
#  Date    : December 5, 2017
#  
#  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#================================ Imports ============================
#%matplotlib inline
import os,cv2
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten

from tensorflow.python.keras.layers import Activation, Dropout


#%%% For error handling "CUBLAS_STATUS_ALLOC_FAILED" on my computer
# Didails=> GitHub Link : https://github.com/tensorflow/tensorflow/issues/7072
# It's work fine now
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#%%%

#% Some helper functions from helper_function.py 
from helper_function import plot_images
from helper_function import plot_miss_classified_images
from helper_function import plot_training_history
from helper_function import plot_confusion_matrix
from helper_function import plot_conv_weights
from helper_function import plot_conv_output

#% Helper functions from loaddataset.py
# import function for generating dataset
from loaddataset import generate_dataset

#=============================================================



#========================== Load Data ========================
# Image size
img_rows=28
img_cols=28
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_rows * img_cols

# Tuple with height and width of images used to reshape arrays.
# This is used for plotting the images.
img_shape = (img_rows,img_cols)

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = (img_rows, img_cols, 1)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channel = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

# Number of epoch 
num_epoch=20

#% set dataset path
# get current directory
PATH = os.getcwd() 
# Define data path
data_path = PATH + '/dataset'
#data_path = 'C:/Users/ASUS/Desktop/dataset'

# Load data
# The returned labels of generate_dataset function is on-hot encoding form
# and shuffled
img_data,labels = generate_dataset(path=data_path, img_shape=img_shape,num_channel=1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(img_data, labels, test_size=0.2, random_state=2)

print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Test-set    :\t\t{}".format(len(y_test)))
#print("- Validation-set:\t{}".format(len(y_validate)))

# test class in integers
test_cls = np.argmax(y_test, axis=1)
#print(test_cls)
#=============================================


#================== Ploat some images ==========
if False:
	# Plot a few images to see if data is correct
	# Get the first images from the test-set.
	images = X_test[0:9]

	# Get the true classes for those images.
	cls_true =np.argmax(y_test[0:9],axis=1)

	# Plot the images and labels
	plot_images(images=images, cls_true=cls_true)

#==============================================



#================= Design Model ===========

# Start construction of the Keras Sequential model.
model = Sequential()

# Add an input layer which is similar to a feed_dict in TensorFlow.
# Note that the input-shape must be a tuple containing the image-size.
# The convolutional layers expect images with shape (28, 28, 1)
model.add(InputLayer(input_shape=img_shape_full))

# First convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=3, strides=1, filters=16, padding='same',
                 activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Second convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=3, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the 4-rank output of the convolutional layers
# to 2-rank that can be input to a fully-connected / dense layer.
model.add(Flatten())

# First fully-connected / dense layer with ReLU-activation.
model.add(Dense(128, activation='relu')) 

# Last fully-connected / dense layer with softmax-activation
# for use in classification.
model.add(Dense(num_classes, activation='softmax'))

#=======================================================


#================== Model Compilation ==================

from tensorflow.python.keras.optimizers import Adam

optimizer = Adam(lr=1e-3)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#=======================================================


#======================= Training Model =================
#batch_size= 128
hist=model.fit(x=X_train,
          y=y_train,
          epochs=num_epoch, batch_size=16,verbose=1, validation_data=(X_test, y_test))

#========================================================


#======================== Plot Training History ===========
if 1:
	# ploting losses and accuracy
	plot_training_history(hist)

#==========================================================


#================= Evaluation of trained model =============
#% Test performance on the test set
result = model.evaluate(x=X_test, y=y_test)

for name, value in zip(model.metrics_names, result):
    print(name, value)

print("The classification accuracy on test set:\n {0}: {1:.2%}".format(model.metrics_names[1], result[1]))


#% Test performance on new images and predict images class
# get current directory
PATH = os.getcwd() 
# Define data path
data_path = PATH + '/validation'
#data_path='C:/Users/ASUS/Desktop/data_git/validation'
#data_path='C:/Users/ASUS/Desktop/NRF_sampol_data'

# load data
val_img_data,val_labels = generate_dataset(path=data_path, img_shape=img_shape,num_channel=1)

images = val_img_data[:]
# convert class labels on-hot encoding to integers
cls_true =np.argmax(val_labels[:],axis=1)
# get the predicted classes as One-Hot encoded arrays
y_pred = model.predict(x=images)
# get the predicted classes as integers
cls_pred = np.argmax(y_pred,axis=1)

if False:
	# plot some images
	plot_images(images=images,
	            cls_true=cls_true,
	            cls_pred=cls_pred,
	            max_im=16)

if True:
	# plot some miss classified images
	mis_clsify = plot_miss_classified_images(images=images,
	                                 cls_true=cls_true,
	                                 cls_pred=cls_pred,
	                                 max_im=16)

	print("Total miss classify:{0} \n Accuracy:{1:.2%}".format( mis_clsify,1-mis_clsify/len(cls_true)))
# accuracy on new images
result = model.evaluate(x=val_img_data,
                        y=val_labels)
print("\nThe classification accuracy on new images:\n {0}: {1:.2%}".format(model.metrics_names[1], result[1]))

#==============================================================



#================== Confusion matrix and Classification report ==========================

#% classification report
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)

y_pred = np.argmax(Y_pred, axis=1)


target_names =  ['class 0(Zero)', 'class 1(One)', 'class 2(Two)','class 3(Three)','class 4(Four)', 'class 5(Five)', 'class 6(Six)','class 7(Seven)', 'class 8(Eight)', 'class 9(Nine)'] 

print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

#% Compute confusion matrix
if True:
	cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

	np.set_printoptions(precision=2)

	plt.figure()

	# Plot non-normalized confusion matrix
	plot_confusion_matrix(cnf_matrix, classes=target_names,
	                      title='Confusion matrix')
	#plt.figure()
	# Plot normalized confusion matrix
	#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
	#                      title='Normalized confusion matrix')
	#plt.figure()
	plt.show()

#%
#===============================================================



#========================= Save Model ==========================

# the file-path where we want to save the Keras model
path_model = 'model.keras'
# save model
model.save(path_model)

#% Saving best weighted model only
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import load_model

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

model.save('model.hdf5')
loaded_model=load_model('model.hdf5')

#====================================================



#=========== Load saved model and test prediction =============

# load saved model as model2
model2 = load_model(path_model)

# test prediction using saved model that is model2
images = X_test[0:9]
cls_true = np.argmax(y_test[0:9],axis=1)
y_pred = model2.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)

if False:
	plot_images(images=images,
	            cls_pred=cls_pred,
	            cls_true=cls_true,
	            max_im=9)

#=====================================================================



#================ Visualization of Layer Weights and Outputs =========

# model summary
model2.summary()

#% Layer Weights
# we have to count the indices to get the layers we want
# The input-layer has index 0
layer_input = model2.layers[0]
# the first convolutional layer has index 1
layer_conv1 = model2.layers[1]
# the second convolutional layer has index 3
layer_conv2 = model2.layers[3]

# now get their weights
weights_conv1 = layer_conv1.get_weights()[0]
# this is a 4-rank tensor
weights_conv1.shape
# plot weights of first convolutional layer
if True:
	plot_conv_weights(weights=weights_conv1, input_channel=0)

# plot weights of second convolutional layer
if False:
	weights_conv2 = layer_conv2.get_weights()[0]
	plot_conv_weights(weights=weights_conv2, input_channel=0)



#% Layer Output 
from tensorflow.python.keras import backend as K

# Visualizing 1st convolutional layer
if True:
	output_conv1 = K.function(inputs=[layer_input.input],
	                          outputs=[layer_conv1.output])

	image1 = X_test[0]
	layer_output1 = output_conv1([[image1]])[0]
	layer_output1.shape
	plot_conv_output(values=layer_output1)

# Visualizing 2nd convolutional layer
if False:
	output_conv2 = K.function(inputs=[layer_input.input],
	                          outputs=[layer_conv2.output])

	layer_output2 = output_conv2([[image1]])[0]
	print(layer_output2.shape)
	plot_conv_output(values=layer_output2)

#===============================================================


#================= The end =====================================

# Now we can delete the model from memory
del model

#===============================================================