from tkinter import Tk, filedialog

import json
import os

from brisk.utils import path

# Set the output base directory and save
# to the config file
def set_directory():
    config_dir = os.path.normpath(os.path.dirname(os.path.dirname(path.__file__)))

    root=Tk()
    root.withdraw()
    root.focus_set()
    output_dir = os.path.normpath(filedialog.askdirectory(parent=root))
    root.destroy()

    config_file_path = os.path.join(config_dir,'config','directories.json')

    conf = {}
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            text = f.read()
            if len(text):
                conf = json.loads(text)

    conf['output_dir'] = os.path.normpath(output_dir)
    with open(config_file_path,'w') as f:
        f.write(json.dumps(conf, indent=4))

    return