from tkinter import Tk, filedialog

import os

# Asks for a folder where the data is stored, 
# as it's saved by the system
def get_folder():
    root=Tk()
    root.withdraw()

    open_file = filedialog.askdirectory()

    root.destroy()

    return open_file

# Gets IMU data filenames from a folder
def get_imu_filename(dirname):

    fnames = os.listdir(dirname)

    return [x for x in fnames if x.startswith('data_')]
