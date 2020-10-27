import tensorflow as tf
import matplotlib.pyplot as plt 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #disable GPU 

batch_size = 128
no_classes = 10
epochs = 3
image_height, image_width = 80, 80

input_shape = (image_height, image_width, 1)


def simple_cnn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(80, 80, 3)
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=no_classes, activation='softmax'))
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
def run_test_harness():
	# define model
	model = simple_cnn()
    # create data generator
	datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = datagen.flow_from_directory('E:\code\AI\Xiangpi-AI-with-board-recognition-via-Rasp-Camera-Module\classify_dog_cat\data/train/',
		class_mode='binary', batch_size=64, target_size=(80, 80))
	test_it = datagen.flow_from_directory('E:\code\AI\Xiangpi-AI-with-board-recognition-via-Rasp-Camera-Module\classify_dog_cat\data/test/',
		class_mode='binary', batch_size=64, target_size=(80, 80))
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=3, verbose=0)
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
run_test_harness()
