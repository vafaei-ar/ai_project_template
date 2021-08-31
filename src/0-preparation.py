from google_drive_downloader import GoogleDriveDownloader as gdd

file_path = 'https://drive.google.com/file/d/1YLj_Kg0z3M3eQSjPyV1Exl1vMVE25XZV/view?usp=sharing'


# 
file_id= '1YLj_Kg0z3M3eQSjPyV1Exl1vMVE25XZV'
gdd.download_file_from_google_drive(file_id=file_id,
                                    dest_path='../data/organmnist_axial.npz',
                                    unzip=False)









