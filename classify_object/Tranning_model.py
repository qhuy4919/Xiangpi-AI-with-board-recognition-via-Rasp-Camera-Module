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

##PARAMATERS##
path = "./classify_object/Trainning_data"
labelFile = './classify_object/Labels.csv'
batch_size_val = 50
steps_per_epoch_val = 200
epochs_val = 20
imageDimesions = (150,150,3)
testRatio = 0.2
validationRatio = 0.2 

count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
noOfClasses=len(myList)
print("Importing Classes.....")
for x in range (0,len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count, end =" ")
    count +=1
images = np.array(images)
classNo = np.array(classNo)

###Split data###
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
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

###CHECK WHETHER INPUT VALID OR NOT 
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
    img = grayscale(img)     # CONVERT TO GRAYSCALE
    img = equalize(img)      # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img
 
X_train=np.array(list(map(preprocessing,X_train)))  # TO IRETATE AND PREPROCESS ALL IMAGES
X_validation=np.array(list(map(preprocessing,X_validation)))
X_test=np.array(list(map(preprocessing,X_test)))
print("ok")
############################### ADD A DEPTH OF 1
X_train= X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation= X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test= X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

#################AUGMENTATAION OF IMAGES: TO MAKEIT MORE GENERIC
dataGen= ImageDataGenerator(width_shift_range=0.1,   # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF PIXELS EG 10 PIXELS
                            height_shift_range=0.1,
                            zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                            shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                            rotation_range=10)  # DEGREES
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20)
X_batch,y_batch = next(batches)

y_train = tf.keras.utils.to_categorical(y_train,noOfClasses)
y_validation = tf.keras.utils.to_categorical(y_validation,noOfClasses)
y_test = tf.keras.utils.to_categorical(y_test,noOfClasses)
################CNN MODEL
def myModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=60,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(150, 150, 1)
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

############################### TRAIN
model = myModel()
model.summary()
model.fit(X_train, y_train, batch_size=batch_size_val, epochs=20, validation_data=(X_test, y_test))
loss,accuracy = model.evaluate(X_test,y_test)
print(loss)
print(accuracy)
# ##########################RESULT
def getCalssName(classNo):
    if   classNo == 0: return 'empty'
    elif classNo == 1: return 'black'
    elif classNo == 2: return 'red'
###################TEST
path_test_folder = "./classify_object/Test_data"
TestList = os.listdir(path_test_folder)
for x in TestList:
    img = cv2.imread(path_test_folder+'/'+x)[:,:,0]
    img = np.asarray(img)
    img = cv2.resize(img, (150, 150))
    img = img.reshape(1, 150, 150, 1)
    prediction = model.predict(img)
    print(f"I guess : {np.argmax(prediction)}")
    print(getCalssName(np.argmax(prediction)))
    
    img_rgb = plt.imread(path_test_folder+'/'+x)
    plt.imshow(img_rgb)
    plt.show()
