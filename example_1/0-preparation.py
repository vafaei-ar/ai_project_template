import numpy as np
from os.path import exists 
from skimage.io import imsave
from google_drive_downloader import GoogleDriveDownloader as gdd

file_path = 'https://drive.google.com/file/d/1YLj_Kg0z3M3eQSjPyV1Exl1vMVE25XZV/view?usp=sharing'


# 
file_id= '1YLj_Kg0z3M3eQSjPyV1Exl1vMVE25XZV'
dest_path='../data/organmnist_axial.npz'

if not exists(dest_path):
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=dest_path,
                                        unzip=False)


data_path = '../data/organmnist_axial.npz'
data = np.load(data_path)

test_images = data['test_images']
inds = np.random.randint(0,test_images.shape[0],5)

for i,ii in enumerate(inds):
    imsave('../data/sample_'+str(i)+'.jpg', test_images[ii])






