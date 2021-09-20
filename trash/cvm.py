import sys 
sys.path.insert(0,'ai_project_template/')

import simpleai as si

# in case this is run outside of conda environment with python2
import os
import sys
import mlflow
import argparse
import numpy as np
import pylab as plt
from glob import glob
from pathlib import Path
import tensorflow as tf
import mlflow.tensorflow
from skimage.io import imread
from skimage.transform import resize
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess(x):
    return si.filters(x,edd_method='sob')


# # Enable auto-logging to MLflow to capture TensorBoard metrics.
# # mlflow.tensorflow.autolog()

mlflow.start_run()

parser = argparse.ArgumentParser()

parser.add_argument("--run_name", default='test', type=str, help="Run name")
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--epochs", default=5, type=int, help="classification epochs")
parser.add_argument("--aug_rot", default=45, type=float, help="augmentation rotation")
parser.add_argument("--aug_w", default=0.05, type=float, help="augmentation width shift")
parser.add_argument("--aug_h", default=0.05, type=float, help="augmentation height shift")
parser.add_argument("--aug_zoom", default=0.05, type=float, help="augmentation zoom")
parser.add_argument("--model_path", default='models/', type=str, help="model path")
parser.add_argument("--data_path", default='CVM/radios_0', type=str, help="data path")

args = parser.parse_args()

run_name = args.run_name
batch_size = args.batch_size
epochs = args.epochs
aug_rot = args.aug_rot
aug_w = args.aug_w
aug_h = args.aug_h
aug_zoom = args.aug_zoom
model_path = args.model_path
data_path = args.data_path

mlflow.set_tag("mlflow.runName", run_name)
Path(model_path).mkdir(parents=True, exist_ok=True)

# run_name = 'test'
# batch_size = 32
# epochs = 50
# aug_rot = 4
# aug_w = 0.02
# aug_h = 0.02
# aug_zoom = 0.03
# model_path = 'models/'
# data_path = 'CVM/radios_0'
n_sample = 4

lx = 256
ly = lx

fname = '{}-{}-{}'.format(data_path.split('/')[-1],lx,ly)

paths = glob(data_path+'/CV*')
assert len(paths)!=0,'No class found!'

if not os.path.isfile(fname+'.npz'):# or restart:
    print("[INFO] reading data and preparation...")
    x = []
    labels = []
    full_path = []
    for path in paths:
        files = glob(path+'/*')
        for fil in files:
#             try:
                img = imread(fil)
#                img = img[:,:1700]
                if lx*ly!=0:
                    img = resize(img,output_shape=(lx,ly))
                if img.ndim==3:
                    img = np.mean(img,axis=-1)
                x.append(img)
                labels.append(fil.split('/')[-2])
                full_path.append(fil)
#             except:
#                 print('Something is wrong with',fil,', skipped.')
    print("[INFO] prepared data is saved.")
    np.savez(fname,x=x,labels=labels,full_path=full_path)
    x = np.array(x)
    labels = np.array(labels)
else:
    data = np.load(fname+'.npz'.format(lx,ly))
    x = data['x']
    labels = data['labels']
    full_path = data['full_path']
    print("[INFO] data is loaded...")

int_map,lbl_map = si.int_label(labels)


# In[10]:


vec = [int_map[word] for word in labels]
x = np.array(x)
vec = np.array(vec)
y = tf.keras.utils.to_categorical(vec, num_classes=None, dtype='float32')
n_data,lx0,ly0 = x.shape
x = x[:,:,:,None]/x.max()

n_class,class_labels, nums = si.describe_labels(y,verbose=0)
n_data = x.shape[0]

x_new = []
for i in range(n_data):
    x_new.append(preprocess(x[i,:,:,0]))
x = np.array(x_new)
n_data,lx0,ly0 = x.shape
x = x[:,:,:,None]/x.max()

print(x.shape,y.shape)

x = x-x.min(axis=(1,2,3),keepdims=1)
x = x/x.max(axis=(1,2,3),keepdims=1)
print(x.shape,y.shape)
print(x.min(),x.max())

x0, y0 = x+0, y+0
x, x_val, y, y_val = train_test_split(x, y, test_size=0.05, random_state=42)

print(x.shape,y.shape)
print(x_val.shape,y_val.shape)

print('[INFO] input/output shapes: {}/{}'.format(x.shape,y.shape))

imgs = []
for i in range(n_class):
    imgs.append(x_val[y_val[:,i].astype(bool)][:n_sample])

imgs = np.array(imgs)
subplot_args = { 'nrows': n_sample, 'ncols': n_class,
                 'figsize': (int(4*n_class),int( 5*n_sample)),
                 'subplot_kw': {'xticks': [], 'yticks': []} }

subplot_args2 = { 'nrows': 1, 'ncols': n_class,
                 'figsize': (int(4*n_class),5),
                 'subplot_kw': {'xticks': [], 'yticks': []} }

f, ax = plt.subplots(**subplot_args)
for i in range(n_class):
  for j in range(n_sample):
    title = lbl_map[i]
    ax[j,i].set_title(title, fontsize=40)
    ax[j,i].imshow(imgs[i,j,:,:,0],cmap='gray')
plt.tight_layout()
figname = model_path+'_samples.jpg'
plt.savefig(figname,dpi=150)
plt.close()
mlflow.log_artifact(figname)

#exit()

# initialize the training data augmentation object
def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 0.01
    deviation = VARIABILITY*np.random.uniform(0,1)
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
#    np.clip(img, 0., 255.)
    return img
    
aug = ImageDataGenerator(rotation_range = aug_rot,
                         width_shift_range = aug_w,
                         height_shift_range = aug_h,
                         zoom_range = aug_zoom,
                         preprocessing_function=add_noise,
                         fill_mode="nearest")

si.describe_labels(y,verbose=1)
x_us,y_us = si.balance_aug(x,y,aug)
# x_us,y_us = mixup(x,y,alpha=20,beta=1)
si.describe_labels(y_us,verbose=1)

x_us,y_us = si.shuffle_data(x_us,y_us)

labeled_train_dataset = aug.flow(x_us,y_us, batch_size=batch_size)
test_dataset = (x_val, y_val)


# y_train = tf.keras.utils.to_categorical(train_labels)
# y_test = tf.keras.utils.to_categorical(val_labels)

# n_class,class_labels, nums = describe_labels(y_train)
# train_images,y_train = balance_aug(train_images,y_train)
# n_class,class_labels, nums = describe_labels(y_train)

shape = x_us.shape[1:]
# uniq_labels = np.unique(train_labels)
#n_class = len(uniq_labels)

loss = tf.keras.losses.CategoricalCrossentropy()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                             initial_learning_rate=1e-3,
                             decay_steps=50,
                             decay_rate=0.95)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

try:
    model = si.model_version(model_path,mode='load')
    assert not model is None
    print('model is loaded!')
except:
    model = si.build_2dcond_model(shape,n_class,
                               n_layers = 4,
                               nch = 8,
                               kernelsize=3,
                               activation="relu",
                               maxpool=0)
    print('training from scratch!')

model.compile(optimizer=opt,
              loss = tf.keras.losses.CategoricalCrossentropy(from_logits=0),
              metrics = [tf.keras.metrics.CategoricalAccuracy(name="acc")],
              )

history = model.fit(labeled_train_dataset,
                    epochs=epochs,
                    validation_data=test_dataset,
                    verbose=0,
                    callbacks=[TqdmCallback(verbose=0),si.MlflowCallback()]
                    )

model_name = si.model_version(model_path,mode='save',model=model)

tags = {'model_path': model_path,
        'model_name': model_name}

# Set a batch of tags
mlflow.set_tags(tags)

# In[14]:


import ktrans as ktr


# In[15]:


ii = np.random.randint(x_val.shape[0])

img = x_val[ii]
lbl = np.argmax(y_val[ii])
methods = [ktr.vanilla_saliency,ktr.smoothgrad,ktr.gradcam,ktr.gradcampp,ktr.scorecam]
m_name = ['vanilla_saliency','smoothgrad','gradcam','gradcampp','scorecam']
nmethods = len(methods)

fig,axs = plt.subplots(1,nmethods,figsize=(5*4,4))

for i in range(nmethods):

    smap = methods[i](img,model,class_id=lbl)

#     axs[0,i].imshow(img)
#     axs[0,i].get_xaxis().set_visible(False)
#     axs[0,i].get_yaxis().set_visible(False)
    
    axs[i].imshow(img)
    axs[i].imshow(smap,cmap='jet',alpha=0.5)
    axs[i].set_title(m_name[i])
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
plt.tight_layout()

figname = model_path+'_xsamples.jpg'
plt.savefig(figname,dpi=150)
plt.close()
mlflow.log_artifact(figname)


# In[16]:


ii = np.random.randint(x_val.shape[0])

for ic in range(n_class):
    filt = np.argmax(y_val,axis=1)==ic
    

    img = x_val[filt][:10]
    lbl = np.argmax(y_val[filt][:10],axis=1)
    methods = [ktr.vanilla_saliency,ktr.smoothgrad,ktr.gradcam,ktr.gradcampp,ktr.scorecam]
    m_name = ['vanilla_saliency','smoothgrad','gradcam','gradcampp','scorecam']
    nmethods = len(methods)

    plt.close()
    fig,axs = plt.subplots(1,nmethods,figsize=(5*4,4))

    for i in range(nmethods):

        smap = methods[i](img,model,class_id=lbl)

    #     axs[0,i].imshow(np.mean(img,axis=0))
    #     axs[0,i].get_xaxis().set_visible(False)
    #     axs[0,i].get_yaxis().set_visible(False)

        axs[i].imshow(np.mean(img,axis=0))
        axs[i].imshow(np.mean(smap,axis=0),cmap='jet',alpha=0.5)
        axs[i].set_title(m_name[i])
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    plt.tight_layout()

    figname = model_path+'_xavg_'+str(ic)+'.jpg'
    plt.savefig(figname,dpi=150)
    plt.close()
    mlflow.log_artifact(figname)
# In[ ]:


mlflow.end_run()

