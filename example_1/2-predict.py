import time
import numpy as np
import streamlit as st
#import cv2
from PIL import Image
#from utils import *
#import models

from utils import model_version

model_path = '../models/'
model = model_version(model_path,mode='load')

if model is None:
    st.write('No model is available!')
    
else:
    uploaded_file = st.file_uploader("Choose a image file for preditcion:",type="jpg")
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = np.array(img)
        print(img.shape)
        print(img.min(),img.max())
        pred = model.predict(img[None,:,:,None])
        pred = np.argmax(pred)
        st.image(img, channels="RGB", caption=['Image label is {}'.format(pred)],use_column_width=1)


