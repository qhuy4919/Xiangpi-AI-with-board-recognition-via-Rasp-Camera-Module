import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt 
import os 
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #disable GPU 

##paramaters##
path = "./Classify/Trainning_data"
labelFile = './Classify/Labels.csv'
batch_size_val = 50
steps_per_epoch_val = 200
epochs_val = 20
imageDimesions = (150,150,3)
testRatio = 1
validationRatio = 1

images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
noOfClasses=len(myList)
print("Importing Classes.....")
for x in range (0,len(myList)):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        images.append(curImg)
        classNo.append(x)
images = np.array(images)
classNo = np.array(classNo)

###Split data###
X_train, X_test, y_train, y_test = train_test_split(images, classNo)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train)
##try to reshpe but dekimasen :(
# X_train = tf.reshape(X_train,[50,50])
# X_test = tf.reshape(X_test,[-1,50,50,1])
# X_validation = tf.reshape(X_validation,[-1,50,50,1])
# #######################
print("Data Shapes")
print("Train",end = "");print(X_train.shape,"-",y_train.shape)
print("Validation",end = "");print(X_validation.shape,"-",y_validation.shape)
print("Test",end = "");print(X_test.shape,"-",y_test.shape)
################## READ CSV FILE
data= pd.read_csv(labelFile,encoding='latin-1',error_bad_lines=False)
print("data shape ",data.shape,type(data))
###check everything 
assert(X_train.shape[0]==y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
assert(X_validation.shape[0]==y_validation.shape[0]), "The number of images in not equal to the number of lables in validation set"
assert(X_test.shape[0]==y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert(X_train.shape[1:]==(imageDimesions))," The dimesions of the Training images are wrong "
assert(X_validation.shape[1:]==(imageDimesions))," The dimesionas of the Validation images are wrong "
assert(X_test.shape[1:]==(imageDimesions))," The dimesionas of the Test images are wrong"




############################### PREPROCESSING THE IMAGES
 
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)     
    img = equalize(img)     
    img = img/255            
    return img
 
X_train=np.array(list(map(preprocessing,X_train)))  # TO IRETATE AND PREPROCESS ALL IMAGES
# X_validation=np.array(list(map(preprocessing,X_validation)))
X_test=np.array(list(map(preprocessing,X_test)))
print("ok")
############################### ADD A DEPTH OF 1
X_train= X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
# X_validation= X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test= X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
#################AUGMENTATAION OF IMAGES: TO MAKEIT MORE GENERIC
dataGen= ImageDataGenerator(width_shift_range=0.1,   # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                            height_shift_range=0.1,
                            zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                            shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                            rotation_range=10)  # DEGREES
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20) 
X_batch,y_batch = next(batches)
################CNN model
def myModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
    model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
    model.add(tf.keras.layers.Dense(2,activation= tf.nn.softmax))
    model.compile(optimizer='Adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    return model

############################### TRAIN
model = myModel()
model.fit(X_train, y_train, epochs=epochs_val)
loss,accuracy = model.evaluate(X_test,y_test)
print(loss)
print(accuracy)

#############Test 
for x in range(0,6):
    img = cv2.imread(f'./Classify/Test_data/{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"I guess the number is: {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
