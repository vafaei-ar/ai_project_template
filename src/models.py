import os
import numpy as np
import pylab as plt
import pandas as pd
from glob import glob
import streamlit as st
import altair as alt
from matplotlib import cm
from skimage.io import imread
from skimage.transform import resize
from scipy.ndimage import gaussian_filter as gauss
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tf_keras_vis.utils import normalize
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import num_of_gpus

from PIL import Image
from pathlib import Path

from utils import *


def data_load(data_path,lx,ly,n_sample,pp,dpi,prefix,restart):

    if pp==0:
        pass
    elif pp==4:
        def preprocess(x):
            return filters(x,edd_method='sob')

    if prefix!='':
        ch_mkdir(prefix)



    if 'npy' in data_path:
        data_set = np.load(data_path,allow_pickle=True)
        data_set.shape = [1]
        data_set = data_set[0]
        x = []
        labels = []
        for key in list(data_set):
            for i in range(len(data_set[key])):
                x.append(data_set[key][i])
                labels.append(key)
        x = np.array(x)
        labels = np.array(labels)
    #    x = np.concatenate(x,axis=0)
    #    labels = np.concatenate(labels,axis=0)
    else:
#        paths = glob(data_path+'/*')
        paths = glob(os.path.join(data_path,'*'))
        print(data_path)
        assert len(paths)!=0,'No data is detected!' 
        fname = '{}-{}-{}'.format(data_path.split('/')[-1],lx,ly)
        if not os.path.isfile(fname+'.npz'):# or restart:
            st.write("[INFO] reading data and preparation...")
            x = []
            labels = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
#            ntot = np.sum([len(glob(path+'/*')) for path in paths])
            ntot = np.sum([len(glob(os.path.join(path,'*'))) for path in paths])
            ip = 0
#            st.write(ntot)
            for path in paths:
#                files = glob(path+'/*')
                files = glob(os.path.join(path,'*'))
                for fil in files:
                    ip += 1
                    progress_bar.progress(ip/ntot)
                    status_text.text('{:4.2f}% complete.'.format(100*ip/ntot))
                    try:
                        img = imread(fil)
        #                img = img[:,:1700]
                        if lx*ly!=0:
                            img = resize(img,output_shape=(lx,ly))
                        if img.ndim==3:
                            img = np.mean(img,axis=-1)
                        x.append(img)
#                        filp = fil.split('/')
                        filp = os.path.normpath(fil)
                        filp = filp.split(os.sep)
                        labels.append(filp[-2])
                    except:
                        print('Something is wrong with',fil,', skipped.')
            st.write("[INFO] prepared data is saved.")
            np.savez(fname,x=x,labels=labels)
            x = np.array(x)
            labels = np.array(labels)
        else:
            data = np.load(fname+'.npz'.format(lx,ly))
            x = data['x']
            labels = data['labels']
            st.write("[INFO] data is loaded...")

    int_map,lbl_map = int_label(labels)

    print(int_map,lbl_map)
#    exit()

    vec = [int_map[word] for word in labels]
    x = np.array(x)
    print(x.shape)

    vec = np.array(vec)
    y = to_categorical(vec, num_classes=None, dtype='float32')
    n_data,lx0,ly0 = x.shape
#    print('XMAX: ',x.max())
    x = x[:,:,:,None]/x.max()

    n_class,class_labels, nums = describe_labels(y,verbose=0)
    
    st.write('Data configuration is:')
#    st.write(class_labels)
#    st.write(lbl_map)
    lbls = [lbl_map[i] for i in range(n_class)]
    st.write(pd.DataFrame(data=np.c_[lbls, class_labels, nums],columns=['labels','index','numbers']))
    
    n_data = x.shape[0]

    x_new = []
    if pp!=0:
        for i in range(n_data):
    #        x[i,:,:,0] = preprocess(x[i,:,:,0])
            x_new.append(preprocess(x[i,:,:,0]))
        x = np.array(x_new)
        n_data,lx0,ly0 = x.shape
        x = x[:,:,:,None]

    x = x-x.min(axis=(1,2,3),keepdims=1)
    x = x/x.max(axis=(1,2,3),keepdims=1)
    print(x.shape,y.shape)
    print(x.min(),x.max())

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
#    plt.savefig(prefix+'samples.jpg',dpi=150)
    plt.savefig(os.path.join(prefix,'samples.jpg'),dpi=150)
    plt.close()
    
    data = [x, x_val, y, y_val]
    return n_class,data

def train(data,n_class,DEEP,EPOCHS,BS,prefix,restart):

    x, x_val, y, y_val = data
    
    n_data,lx0,ly0,_ = x.shape
    # initialize the training data augmentation object
    trainAug = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.03,
        height_shift_range=0.03,
    #   brightness_range=0.01,
    #   shear_range=0.0,
        zoom_range=0.03,
    #   horizontal_flip=True,
    #   vertical_flip=True,
        fill_mode="nearest")
    describe_labels(y,verbose=1)
    x_us,y_us = balance_aug(x,y,trainAug)
    # x_us,y_us = mixup(x,y,alpha=20,beta=1)
    describe_labels(y_us,verbose=1)

    x_us,y_us = shuffle_data(x_us,y_us)

    tf.keras.backend.clear_session()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                 initial_learning_rate=1e-3,
                                 decay_steps=100,
                                 decay_rate=0.95)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss = tf.keras.losses.CategoricalCrossentropy()

    # model.reset_states() 

    if not os.path.isfile(prefix+'model.h5') or restart:
        st.write('[INFO] building neural net...')
        # initialize the initial learning rate, number of epochs to train for,
        # and batch size

        model = build_2dcond_model(shape=(lx0,ly0,1),
                                   n_class=n_class,loss=loss,opt=opt,
                                   n_layers = DEEP,
                                   nch = 8,
                                   kernelsize=3,
                                   activation="relu",
                                   maxpool=2)

        # train the head of the network
        st.write("[INFO] training ...")

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        col1, col2 = st.beta_columns(2)

        ip = 0
        for epoch in range(EPOCHS):
            ip = epoch+1
            progress_bar.progress(ip/EPOCHS)
            status_text.text('{:4.2f}% complete.'.format(100*ip/EPOCHS))

            H = model.fit(trainAug.flow(x_us, y_us, batch_size=BS),
	                  steps_per_epoch=len(x) // BS,
	                  epochs=1,
	                  validation_data=(x_val, y_val),
    # 	              callbacks=callbacks,
                      verbose=0)

            acc1 = H.history['accuracy'][0]
            acc2 = H.history['val_accuracy'][0]
            lss1 = H.history['loss'][0]
            lss2 = H.history['val_loss'][0]
     
            with col1:
                df_acc = pd.DataFrame(columns=['train','valid'],data=[[acc1,acc2]])
                if epoch==0:
                    chart1 = st.line_chart(df_acc)
                    col1.write('Accuracy')
                else:
                    chart1.add_rows(df_acc)
            with col2:
                df_lss = pd.DataFrame(columns=['train','valid'],data=[[lss1,lss2]])
                if epoch==0:
                    chart2 = st.line_chart(df_lss)
                    col2.write('loss')
                else:
                    chart2.add_rows(df_lss)

#        H = model.fit(trainAug.flow(x_us, y_us, batch_size=BS),
#	                  steps_per_epoch=len(x) // BS,
#	                  epochs=EPOCHS,
#	                  validation_data=(x_val, y_val),
#    # 	              callbacks=callbacks,
#                      verbose=0)


#        model.save(prefix+'model.h5', save_format="h5")
        model.save(os.path.join(prefix,'model.h5'), save_format="h5")
        
        plt.plot(H.history['accuracy'])
        plt.plot(H.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
    #    plt.legend(['train', 'val'], loc='upper left')
#        plt.savefig(prefix+'train.jpg')
        plt.savefig(os.path.join(prefix,'train.jpg'),dpi=150)
        plt.close()
        
        st.write('[INFO] trained model is saved.')
#        st.balloons()

    else:
    #_, gpus = num_of_gpus()
    #print('{} GPUs'.format(gpus))
#        model = load_model(prefix+'model.h5')
        model = load_model(os.path.join(prefix,'model.h5'))
        st.write('[INFO] trained model is loaded.')
        print('[INFO] model summary is:\n\n')
        model.summary()
        print('\n\n\n')


    









