import os
import cv2
import mlflow
import numpy as np
import pylab as plt
from glob import glob
import streamlit as st
from pathlib import Path
from datetime import datetime
from subprocess import getoutput

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import linear
from tensorflow.keras.utils import to_categorical
from streamlit.report_thread import get_report_ctx
from tensorflow.keras.callbacks import LearningRateScheduler

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_gpu_memory():
    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        memory_free_values = memory_free_values[0]
    except:
        memory_free_values = 0
    return memory_free_values

def model_version(path,mode,model=None,df=None,prefix=''):
  hist = glob(path+'/*.h5')
  
  if hist==[]:
    last_ind = -1
  else:
    inds = [int(i.split('v')[-1].split('_')[0]) for i in hist]
    last_ind = np.max(inds)
  
  if mode=='save':
    assert not model is None,'model is not given!'
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    last_ind += 1
    model_name = '{}/v{}_{}.h5'.format(path+prefix,last_ind,dt_string)
    # np.save(model_name,[last_ind])
    model.save(model_name)
    if not df is None:
      df_name = '{}/v{}_{}.csv'.format(path+prefix,last_ind,dt_string)
      df.to_csv(df_name)

#    fid = getoutput("xattr -p 'user.drive.id' '{}.npy'".format(model_name))
    return model_name

  elif mode=='load':
    if last_ind==-1:
      return None
    # model = np.load(hist[last_ind])
    model = tf.keras.models.load_model(hist[last_ind])

    return model
  else:
    assert 0,'mode error!'

class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.

        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'

        """
        for key, val in kwargs.items():
            setattr(self, key, val)


#@st.cache(allow_output_mutation=True)
#def get_session(id, **kwargs):
#    return SessionState(**kwargs)


#def get(**kwargs):
#    """Gets a SessionState object for the current session.

#    Creates a new object if necessary.

#    Parameters
#    ----------
#    **kwargs : any
#        Default values you want to add to the session state, if we're creating a
#        new one.

#    Example
#    -------
#    >>> session_state = get(user_name='', favorite_color='black')
#    >>> session_state.user_name
#    ''
#    >>> session_state.user_name = 'Mary'
#    >>> session_state.favorite_color
#    'black'

#    Since you set user_name above, next time your script runs this will be the
#    result:
#    >>> session_state = get(user_name='', favorite_color='black')
#    >>> session_state.user_name
#    'Mary'

#    """
#    ctx = get_report_ctx()
#    id = ctx.session_id
#    return get_session(id, **kwargs)


#def ch_mkdir(directory):
#    Path(directory).mkdir(parents=True, exist_ok=True)
##    if not os.path.exists(directory):
##        try:
##            os.makedirs(directory)
##        except:
##            print('could not make the directory!')

def describe_labels(y0,verbose=0):
    y = y0+0
    if y.ndim==2:
        y = np.argmax(y,axis=1)
    class_labels, nums = np.unique(y,return_counts=True)
    n_class = len(class_labels)
    if verbose:
        print('labels/numbers are:\n',*['{:5s}/{:6d}\n'.format(str(i),j) for i,j in zip(class_labels,nums)])
    return n_class,class_labels, nums

def augment(aug,x):
    aug.fit(x)
    out = []
    for i in x:
        out.append(aug.random_transform(i))
    return np.array(out)

def balance_aug(x0,y0,aug=None,mixup=False):
    x = x0+0
    y = y0+0
    n_class,class_labels, nums = describe_labels(y,verbose=0)
    nmax = max(nums)
    for i,(lbl,n0) in enumerate(zip(class_labels,nums)):
        if nmax==n0:
            continue
        delta = nmax-n0
        if y.ndim==1:
            filt = y==lbl
        elif y.ndim==2:
            filt = y[:,i].astype(bool)
        else:
            assert 0,'Unknown label shape!'
        x_sub = x[filt]
        y_sub = y[filt]
        inds = np.arange(n0)
        nrep = (nmax//len(inds))+1
        inds = np.repeat(inds, nrep)
        np.random.shuffle(inds)
        inds = inds[:delta]
        x_sub = x_sub[inds]
        y_sub = y_sub[inds]
        if not aug is None:
            x_sub = augment(aug,x_sub)
        x = np.concatenate([x,x_sub],axis=0)
        y = np.concatenate([y,y_sub],axis=0)
    return x,y

def mixup(x0,y0,alpha,beta,num_classes=None):
    x = x0+0
    y = y0+0
    
    tocat = False
    if y.ndim==1:
        y = to_categorical(y,num_classes=num_classes)
        tocat = True
        print('The labels are converted into categorical')
        
    n_class,class_labels, nums = describe_labels(y,verbose=0)
    nmax = max(nums)
    # for i,(lbl,n0) in enumerate(zip(class_labels,nums)):
    for i,n0 in enumerate(nums):

        if nmax==n0 or n0==0:
            continue
        delta = int(nmax-n0)
        
        x_sub = x[y[:,i].astype(bool)]
        y_sub = y[y[:,i].astype(bool)]

        inds = np.arange(n0)
        nrep = (nmax//len(inds))
        inds = np.repeat(inds, nrep)
        np.random.shuffle(inds)
        inds = inds[:delta].astype(int)

        x_sub = x_sub[inds]
        y_sub = y_sub[inds]

        b = np.random.beta(alpha,beta,delta)[:,None]

        inds = np.arange(x.shape[0])
        np.random.shuffle(inds)
        inds = inds[:delta]
        xt = x[inds]
        yt = y[inds]

        if x.ndim==2:
            x_sub = b[:,:]*x_sub+(1-b[:,:])*xt
        elif x.ndim==3:
            x_sub = b[:,:,None]*x_sub+(1-b[:,:,None])*xt
        elif x.ndim==4:
            x_sub = b[:,:,None,None]*x_sub+(1-b[:,:,None,None])*xt
        else:
            assert 0,'The shape is not as expected! {}-{}'.format(x.shape,x_sub.shape)
        
        y_sub = b*y_sub+(1-b)*yt

        x = np.concatenate([x,x_sub],axis=0)
        y = np.concatenate([y,y_sub],axis=0)
    return x,y

class DataFeed:
    def __init__(self,x,y,aug = None):
        self.x = x
        self.y = y
        self.aug = aug
        self.nd,self.nx,self.ny,self.ch = x.shape
        self.banance()
        
    def banance(self):
        self.xb,self.yb = balance_aug(self.x,self.y,aug=self.aug)
        self.ndb = self.xb.shape[0]
    def __call__(self,num,reset=False):
        if reset:
            self.banance()
        inds = np.arange(self.ndb)
        np.random.shuffle(inds)
        inds = inds[:num]
        return self.xb[inds],self.yb[inds]

def shuffle_data(x,y):
    ndata = x.shape[0]
    inds = np.arange(ndata)
    np.random.shuffle(inds)
    return x[inds],y[inds]
    

def int_label(labels):
    dummy = {j: i for i,j in enumerate(labels)}
    int_map = {}
    lbl_map = {}
    for j,(i,_) in enumerate(dummy.items()):
        int_map[i] = j
        lbl_map[j] = i
    return int_map,lbl_map

class MlflowCallback(tf.keras.callbacks.Callback):
    
    # This function will be called after each epoch.
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        # Log the metrics from Keras to MLflow     
        mlflow.log_metric('loss', logs['loss'], step=epoch)
        mlflow.log_metric('val_loss', logs['val_loss'], step=epoch)
        mlflow.log_metric('acc', logs['acc'], step=epoch)
        mlflow.log_metric('val_acc', logs['val_acc'], step=epoch)
        
        
        for il,l in enumerate(self.model.layers):
            try:
                zf = tf.math.zero_fraction(l.weights[0]).numpy()
                mlflow.log_metric(l.name+'_zero_frac', zf, step=epoch)
            except:
                pass
    
    # This function will be called after training completes.
    def on_train_end(self, logs=None):
        mlflow.log_param('num_layers', len(self.model.layers))
        mlflow.log_param('optimizer_name', type(self.model.optimizer).__name__)

def build_2dcond_model(shape,n_class,
                       loss=None,opt=None,
                       n_layers = 1,
                       nch = 8,
                       kernelsize=3,
                       activation="relu",
                       maxpool=0,
                       metrics=["accuracy"]):
    
    inp = layers.Input(shape=shape, name="img")
    xl = layers.Conv2D(nch, kernelsize, activation=activation)(inp)
    
    for _ in range(n_layers//2):
        nch *= 2
        xl = layers.Conv2D(nch, kernelsize, activation=activation)(xl)
        xl = layers.Conv2D(nch, kernelsize, activation=activation)(xl)
        if maxpool>1:
            xl = layers.MaxPooling2D(maxpool)(xl)
                           
    for _ in range(n_layers//2):
        nch /= 2
        xl = layers.Conv2D(nch, kernelsize, activation=activation)(xl)
        xl = layers.Conv2D(nch, kernelsize, activation=activation)(xl)
        if maxpool>1:
            xl = layers.MaxPooling2D(maxpool)(xl)

    xl = layers.Conv2D(nch, kernelsize, activation=activation)(xl)
    xl = layers.Flatten(name="flatten")(xl)
    xl = layers.Dense(32, activation="relu")(xl)
    xl = layers.Dropout(0.5)(xl)
    yl = layers.Dense(n_class, activation="softmax")(xl)

    model = Model(inputs=inp, outputs=yl)
    model.summary()
    if not loss is None and not opt is None:
        model.compile(loss=loss, optimizer=opt,metrics=metrics)
    return model

def loss(output):
    return output

def loss_maker(i):
    def loss(output):
        return output[:,i]
    return loss

def model_modifier(m):
    m.layers[-1].activation = linear
    return m


def filters(d,edd_method='sch',R=0,smoothing='g'):

    if (R!=0):
        dt = np.fft.fft2(d)
        if smoothing=='g':
            for i in range(sz):
                for j in range(sz):
                    k2 = 1.*(i*i+j*j)/d.shape[0]
                    dt[i,j]=dt[i,j]*np.exp(-k2*R*R/2)

        elif smoothing=='tp':
            for i in range(sz):
                for j in range(sz):
                    k = np.sqrt(0.001+i*i+j*j)/sz
                    dt[i,j]=dt[i,j]* 3*(np.sin(k*R)-k*R*np.cos(k*R))/(k*R)**3

        d = np.fft.ifft2(dt)
        d = abs(d)

    if edd_method=='lap':
        d = cv2.Laplacian(d,cv2.CV_64F)

    elif edd_method=='sob':
        sobelx = cv2.Sobel(d,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(d,cv2.CV_64F,0,1,ksize=3)
        d =np.sqrt(sobelx**2+sobely**2)

    elif edd_method=='sch':
        scharrx = cv2.Scharr(d,cv2.CV_64F,1,0)
        scharry = cv2.Scharr(d,cv2.CV_64F,0,1)
        d =np.sqrt(scharrx**2+scharry**2)
        
    elif edd_method=='der':
        (dx,dy,vx,vy) = myr.vdd(d)
        d = np.sqrt(dx**2+dy**2)
        
    else:
        print('The filter name is not recognized!')
        
    return d









