
import os

# If you dont want to use gpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import time
import numpy as np
import streamlit as st
import cv2
from skimage.transform import resize
from tf_keras_vis.utils import normalize
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.scorecam import ScoreCAM
from tf_keras_vis.gradcam import Gradcam,GradcamPlusPlus
from scipy.ndimage import gaussian_filter as gauss
from utils import *
import models
import easygui

#st.set_page_config(layout="wide")

session_state = get(dirname=None,reset_model=True)

#progress_bar = st.sidebar.progress(0)
#status_text = st.sidebar.empty()
#last_rows = np.random.randn(1, 1)
#chart = st.line_chart(last_rows)

#for i in range(1, 101):
#    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
##    status_text.text("%i%% Complete" % i)
#    chart.add_rows(new_rows)
##    progress_bar.progress(i)
#    last_rows = new_rows
#    time.sleep(0.05)

##progress_bar.empty()

## Streamlit widgets automatically run the script from top to bottom. Since
## this button is not connected to any other logic, it just causes a plain
## rerun.
#st.button("Rerun!")

#my_bar = st.progress(0)
#for percent_complete in range(100):
#    time.sleep(0.1)
#    my_bar.progress(percent_complete + 1)
#    st.write('{}%'.format((percent_complete+1)/100))


def train():

#    data_path_opt = args.dataset
#    data_path_opt = st.text_input('Enter dataset path:', value='dataset/')
    
#    # import libraries
#    import tkinter as tk
#    from tkinter import filedialog

#    # Set up tkinter
#    root = tk.Tk()
#    root.withdraw()

#    # Make folder picker dialog appear on top of other windows
#    root.wm_attributes('-topmost', 1)


    

    # Folder picker button
    st.write('Please select the dataset directory:')
    clicked = st.button('Selected folder')
    dirname = None
    if clicked:
        dirname = easygui.diropenbox(title='dataset')
#        dirname = st.text_input('Selected folder:', dirname)
#        filedialog.askdirectory(master=root))
        session_state.dirname = dirname
#    st.subheader('You selected comedy.')
#    if st.button('Upload file'):  
#        
#        print(easygui.fileopenbox())
    
#    data_path = session_state.dirname
    data_path = st.text_input('Selected model file:', session_state.dirname)
    
    model_opt = st.selectbox('Please select the architecture?',
                             ('Model 1', 'Model 2'))
    lx = st.number_input('Image size', value=64)
    EPOCHS = st.number_input('Number of epochs', value=10)

    n_sample = 3
    restart = 1
#    EPOCHS = 2
    BS = 4

#    data_path = data_path_opt
    ly = lx                         
    pp = 4
    dpi = 150
    prefix = ''


    if model_opt=='Model 1':
        DEEP = 2
    elif model_opt=='Model 2':
        DEEP = 3
        
    if st.button('Train'):
        if data_path is None:
            st.write('You need to set the data directory first!')
        else:
        
            n_class,data = models.data_load(data_path,lx,ly,n_sample,pp,dpi,prefix,restart)
            models.train(data,n_class,DEEP,EPOCHS,BS,prefix,restart)
        
#            models.train(data_path,DEEP,EPOCHS,BS,lx,ly,n_sample,pp,dpi,prefix,restart)
#            session_state.reset_model
    
#        st.write('Training...')
#        progress_bar = st.progress(0)
#        status_text = st.empty()
#        chart = st.line_chart([1.])

#        for i in range(100):
#            # Update progress bar.
#            progress_bar.progress(i + 1)

#            new_rows = np.random.randn(1, 2)

#            # Update status text.
#            status_text.text('{}%'.format(100*(i+1)/100))

#            # Append data to the chart.
#            chart.add_rows([1/(i+1)])

#            # Pretend we're doing some computation that takes time.
#            time.sleep(0.02)

#        status_text.text('Done!')
        
        
        
    return

def predict():

    cmap = plt.get_cmap('jet')

    lx,ly = 256,256
    int_map = {1: '06-class__2', 0: '04-class__1'}
#    {'04-class__1 EO': 0, '06-class__2 EO': 1} {0: '04-class__1 EO', 1: '06-class__2 EO'}

#    model_file = st.text_input('Enter model file:', value='none')
    st.write('Selected model file:')
    clicked = st.button('Model')
    if clicked:
        dirname = easygui.fileopenbox(title='model selection',filetypes=['*.h5'])
#        filedialog.askdirectory(master=root))
        session_state.dirname = dirname

#    st.subheader('You selected comedy.')
#    if st.button('Upload file'):  
#        
#        print(easygui.fileopenbox())
#    model_file = session_state.dirname
    model_file = st.text_input('Selected model file:', session_state.dirname)
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

    option = st.selectbox(
        'How would you like to choose as the attention analyzer?',
        ('Filter', 'Saliency', 'SmoothGrad', 'GradCAM', 'GradCAM++', 'ScoreCAM'))


    towcol = 1

    if uploaded_file is not None and model_file != 'Filter':
        # Convert the file to an opencv image.
        model = load_model(model_file)
        _,lx,ly,_ = model.layers[0].input_shape[0]
        
        if option=='Saliency':  
            ## # Vanilla Saliency
            saliency = Saliency(model,
                                model_modifier=model_modifier,
                                clone=False)
        
        if option=='SmoothGrad':
            ## # SmoothGrad
            saliency = Saliency(model,
                                model_modifier=model_modifier,
                                clone=False)        
        
        if option=='GradCAM':    
            gradcam = Gradcam(model,
                              model_modifier=model_modifier,
                              clone=False)       
        
        if option=='GradCAM++':
            gradcam = GradcamPlusPlus(model,
                                      model_modifier=model_modifier,
                                      clone=False)
        if option=='ScoreCAM':
            scorecam = ScoreCAM(model, model_modifier, clone=False)
            
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        img = cv2.imdecode(file_bytes, 1)
        img = resize(img,output_shape=(lx,ly))
        if img.ndim==3:
            img = np.mean(img,axis=-1)
        
        pp = filters(img,edd_method='sob')
        pp = pp-pp.min()
        pp = pp/pp.max()
        pp = pp[None,:,:,None]
        
#        st.image(pp,use_column_width=1)
#        st.write(pp.min(),pp.max())
        y_p = model.predict(pp)
        pind = np.argmax(y_p)
        st.markdown('**_CLASS_ {}**'.format(pind))

        if option=='Filter':
            att = pp[0,:,:,0]

        if option=='Saliency':
            loss = loss_maker(pind)
            att = saliency(loss,pp)[0]
            att = normalize(att)

        if option=='SmoothGrad':
            loss = loss_maker(pind)
            att = saliency(loss,pp,
                            smooth_samples=20,
                            smooth_noise=0.20)[0]
            att = normalize(att)

        if option=='GradCAM':
            loss = loss_maker(pind)
            att = gradcam(loss,pp,penultimate_layer=-1)[0]
            att = normalize(att)

        if option=='GradCAM++':
            loss = loss_maker(pind)
            att = gradcam(loss,pp,penultimate_layer=-1)[0]
            att = normalize(att)

        if option=='ScoreCAM':
            loss = loss_maker(pind)
            att = scorecam(loss,pp,penultimate_layer=-1)[0]
            att = normalize(att)

        if option!='Filter':
            att = gauss(att,3)
        att = cmap(att)
        att = att/att.max()
#        lbl = int_map[pind]
        
        img = img[:,:,None]
        img = np.concatenate(3*[img]+[np.ones(img.shape)],axis=-1)
        att[:,:,:3] = 0.7*img[:,:,:3]+0.3*att[:,:,:3]
        
        if towcol:
            col1, col2 = st.beta_columns(2)
    #        with col1:
            col1.image(img, channels="RGB", caption=['Image is {}'.format(pind)],use_column_width=1)
    #        with col2:
            col2.image(att, channels="RGB", caption=['Attention'],use_column_width=1)
    #        
        else:
            showatt = st.checkbox('show attention')
            if showatt:
                st.image(att, channels="RGB", caption=['Attention'],use_column_width=1)
            else:
                st.image(img, channels="RGB", caption=['Image is {}'.format(pind)],use_column_width=1)
        
        # Now do something with the image! For example, let's display it:
    #    st.image([img,pred], channels="BGR", width=300, caption=['Image','Attention map'])
    return


st.title('Model training and prediction application.')

mode = st.radio(
    "Please choose the procedure:",
    ('Train a model', 'Predict with a model'))
if mode == 'Train a model':
    st.subheader('You are going to train a model.')
    train()
elif mode == 'Predict with a model':
    st.subheader("You are going to predict with a trained model.")
    predict()





















