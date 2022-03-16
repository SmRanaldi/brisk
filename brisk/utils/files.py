from tkinter import Tk, filedialog

# Asks for a folder where the data is stored, 
# as it's saved by the system
def load_folder():
    root=Tk()
    root.withdraw()

    open_file = filedialog.askdirectory()

    root.destroy()

    return open_file