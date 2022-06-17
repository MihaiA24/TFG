from six import b
import tensorflow as tf
from datasetV2 import *
from modelv2 import UNET
from model3 import simple_unet_model_with_jacard
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

# path

train_path =  'train_dataset.csv' 
val_path =   'val_dataset.csv' 

SEED = 42
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 144
AUTOTUNE = tf.data.experimental.AUTOTUNE


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)



# set train and val dataset
BATCH_SIZE = 8 
BUFFER_SIZE = 1000

# load train dataset
train_images, train_masks = load_data(train_path)
train_dataset = tf_dataset(train_images, train_masks,batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
# load val dataset
val_images, val_masks = load_data(val_path)
val_dataset = tf_dataset(val_images, val_masks, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)

# model = UNET(n_classes=1, IMG_HEIGHT=224, IMG_WIDTH=144, IMG_CHANNELS=3)
model = UNET(input_size=(224, 144, 3), n_filters=2, n_classes=1)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.Adam(), 
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# Model V3
# model = simple_unet_model_with_jacard(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
model.summary()

TRAINSET_SIZE = len(train_images)
VALSET_SIZE = len(val_images)
print('Train size: ',TRAINSET_SIZE)
print('Val size: ',VALSET_SIZE)



EPOCHS = 10
STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE
with tf.device("/gpu:0"):
    history = model.fit(train_dataset, epochs=EPOCHS,batch_size=BATCH_SIZE,
                                steps_per_epoch=STEPS_PER_EPOCH,
                                validation_steps=VALIDATION_STEPS,
                                validation_data=val_dataset)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()