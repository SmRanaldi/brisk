from tkinter import Tk, filedialog

import os

# Asks for a folder where the data is stored, 
# as it's saved by the system
def get_folder():
    root=Tk()
    root.withdraw()
    root.focus_set()

    open_file = filedialog.askdirectory(parent=root)

    root.destroy()

    return open_file

# Gets IMU data filenames from a folder
def get_imu_filename(dirname):

    fnames = os.listdir(dirname)

    return [x for x in fnames if x.startswith('data_')]

# Join path names with a forward slash
# for cross-platform support
def join_path(components):

    return os.path.normpath(os.path.join(*components))