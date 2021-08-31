# in case this is run outside of conda environment with python2
import sys
import mlflow
import argparse
import numpy as np
import tensorflow as tf
import mlflow.tensorflow
from tqdm.keras import TqdmCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import *

# Enable auto-logging to MLflow to capture TensorBoard metrics.
# mlflow.tensorflow.autolog()

parser = argparse.ArgumentParser()

parser.add_argument("--run_name", default='test', type=str, help="Run name")
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--epochs", default=5, type=int, help="classification epochs")
parser.add_argument("--aug_rot", default=45, type=float, help="augmentation rotation")
parser.add_argument("--aug_w", default=0.05, type=float, help="augmentation width shift")
parser.add_argument("--aug_h", default=0.05, type=float, help="augmentation height shift")
parser.add_argument("--aug_zoom", default=0.05, type=float, help="augmentation zoom")
parser.add_argument("--model_path", default='../models/', type=str, help="model path")

args = parser.parse_args()

run_name = args.run_name
batch_size = args.batch_size
epochs = args.epochs
aug_rot = args.aug_rot
aug_w = args.aug_w
aug_h = args.aug_h
aug_zoom = args.aug_zoom
model_path = args.model_path

mlflow.start_run()
mlflow.set_tag("mlflow.runName", run_name)

data_path = '../data/organmnist_axial.npz'
data = np.load(data_path)

for i in list(data):
    exec("{}=data['{}']".format(i,i))

train_images = train_images[...,None]
val_images = val_images[...,None]
test_images = test_images[...,None]

print(train_images.shape,val_images.shape,test_images.shape)
print(train_labels.shape,val_labels.shape,test_labels.shape)

aug = ImageDataGenerator(rotation_range = aug_rot,
                         width_shift_range = aug_w,
                         height_shift_range = aug_h,
                         zoom_range = aug_zoom,
                         fill_mode="nearest")

y_train = tf.keras.utils.to_categorical(train_labels)
y_test = tf.keras.utils.to_categorical(val_labels)

n_class,class_labels, nums = describe_labels(y_train)
train_images,y_train = balance_aug(train_images,y_train)
n_class,class_labels, nums = describe_labels(y_train)

shape = train_images.shape[1:]
uniq_labels = np.unique(train_labels)
#n_class = len(uniq_labels)

loss = tf.keras.losses.CategoricalCrossentropy()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                             initial_learning_rate=1e-3,
                             decay_steps=50,
                             decay_rate=0.95)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model = build_2dcond_model(shape,n_class,
                           n_layers = 4,
                           nch = 8,
                           kernelsize=3,
                           activation="relu",
                           maxpool=0)

labeled_train_dataset = aug.flow(train_images, y_train, batch_size=batch_size)
test_dataset = (val_images, y_test)

model.compile(optimizer=opt,
              loss = tf.keras.losses.CategoricalCrossentropy(from_logits=0),
              metrics = [tf.keras.metrics.CategoricalAccuracy(name="acc")],
              )

history = model.fit(labeled_train_dataset,
                    epochs=epochs,
                    validation_data=test_dataset,
                    verbose=0,
                    callbacks=[TqdmCallback(verbose=0),MlflowCallback()]
                    )

model_name = model_version(model_path,mode='save',model=model)

tags = {'model_path': model_path,
        'model_name': model_name}

# Set a batch of tags
mlflow.set_tags(tags)

mlflow.end_run()


