import os
import site
import threading
import time
import glob
import numpy as np
import tkinter as tk
from tkinter import HORIZONTAL, VERTICAL, Tk, Text, PhotoImage, Canvas, NW
from tkinter.ttk import Progressbar, Button, Label, Style, Scrollbar
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import messagebox
from queue import Empty
from PIL import ImageTk, Image
from torchvision.transforms import Resize

from medseg.pipeline import pipeline
from medseg.monitoring import get_stats
from medseg.utils import DummyTkIntVar

if os.name == "nt":
    ICON_PNG = os.path.join(site.getsitepackages()[1], "medseg", "icon.png")
else:
    ICON_PNG = os.path.join(site.getsitepackages()[0], "medseg", "icon.png")
DEFAULT_TITLE = "Multitasking Lung and findings segmentation on chest CT of COVID patients"


# Simple GUI utils
def file_dialog(dir=False):
    '''
    Simple GUI to chose files
    '''
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    if dir:
        filename = askdirectory()
    else:
        filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    if filename:
        return filename
    else:  # if empty return None
        return None


def alert_dialog(msg, title=DEFAULT_TITLE):
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    messagebox.showinfo(title, msg)


def error_dialog(msg, title=DEFAULT_TITLE):
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    messagebox.showerror(title, msg)


def confirm_dialog(msg, title=DEFAULT_TITLE):
    '''
    Simple confirmation dialog
    '''
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    MsgBox = messagebox.askquestion(title, msg)
    if MsgBox == 'yes':
        return True
    else:
        return False


class MainWindow(threading.Thread):
    def __init__(self, args, info_q):
        '''
        Info_q: queue communication highway with the external world (mainly worker threads)
        '''
        super().__init__()
        self.args = args
        self.info_q = info_q
        self.runlist = None
        self.pipeline = None
        self.resizer = Resize((128, 128))
        self.cli = self.args.input_folder is not None and self.args.output_folder is not None
        if self.cli:
            print("Input and output given through CLI, not invoking GUI.")
            assert os.path.exists(self.args.input_folder), f"Input folder {self.args.input_folder} doesn't exist."
            self.input_path = self.args.input_folder
            self.display = DummyTkIntVar(value=0)
            self.long = DummyTkIntVar(value=0)
            self.general_progress = {}
            self.iter_progress = {}
            self.cli_populate_runlist()

        self.start()
        
    def start_processing(self):
        '''
        Keep loop ready to receive strings and bar increase information from outside (infoq)
        '''
        if self.runlist is None:
            self.write_to_textbox("\nPlease load a folder or image before starting processing.\n")
            return
        
        if self.args.output_folder is None:
            alert_dialog("Double click (enter) the folder where you want to save the segmentation output.")
            output_folder = None
            while output_folder is None:
                output_folder = file_dialog(dir=True)
            self.write_to_textbox(f"\nWill save outputs in {output_folder}.\n")
        else:
            output_folder = self.args.output_folder

        # Separate thread for heavy processing. Threads for using less ram. Multiprocessing might be faster.
        self.pipeline = threading.Thread(target=pipeline, args=(self.args.model_path, 
                                                                self.runlist, 
                                                                self.args.batch_size, 
                                                                output_folder,
                                                                bool(self.display.get()),
                                                                self.info_q,
                                                                self.args.cpu,
                                                                self.args.win_itk_path,
                                                                self.args.linux_itk_path,
                                                                self.args.debug,
                                                                bool(self.long.get()),
                                                                self.args.atm_mode))                                                          
        self.pipeline_comms_thread = threading.Thread(target=self.pipeline_comms)                                                                
        self.pipeline_comms_thread.start()
        self.pipeline.start()

    def display_slice(self, slice):
        if not self.cli:
            pil_image = self.resizer(Image.fromarray((slice*255).astype(np.uint8)))
            self.img = ImageTk.PhotoImage(pil_image)
            self.canvas.create_image(0, 0, anchor=NW, image=self.img)

    def pipeline_comms(self):
         while True:
            try:
                info = self.info_q.get()
                if info is None:
                    self.write_to_textbox("Closing worker thread...")
                    self.pipeline.join()
                    self.write_to_textbox("Done.")
                    self.runlist = None
                    self.pipeline = None
                    self.set_icon()
                    return
                else:
                    try:
                        if info[0] == "write":
                            self.write_to_textbox(str(info[1]))
                        elif info[0] == "iterbar":
                            self.iter_progress['value'] = int(info[1])
                        elif info[0] == "generalbar":
                            self.general_progress['value'] = int(info[1])
                        elif info[0] == "slice":
                            self.display_slice(info[1])
                        elif info[0] == "icon":
                            self.set_icon()
                    except Exception as e:
                        self.write_to_textbox(f"Malformed pipeline message: {e}. Please create an issue on github.")
                        quit()
            except Empty:
                pass

    def populate_runlist(self):
        self.general_progress['value'] = 0
        self.iter_progress['value'] = 0

        if self.input_path is None:
            pass
        elif os.path.exists(self.input_path) and (".nii" in self.input_path or os.path.isdir(self.input_path) or ".dcm" in self.input_path):
            if os.path.isdir(self.input_path):
                self.write_to_textbox(f"Searching {self.input_path} for files...")
                self.runlist = glob.glob(os.path.join(self.input_path, "*.nii")) + glob.glob(os.path.join(self.input_path, "*.nii.gz"))
                if len(self.runlist) == 0:
                    self.write_to_textbox("Did not find NifT files. Looking for .dcm series...")
                    self.runlist = glob.glob(os.path.join(self.input_path, "*.dcm"))
                    dcms = [os.path.basename(x) for x in self.runlist]
                    if len(self.runlist) > 0:
                        self.write_to_textbox(f"Found {dcms} DCM inside folder {self.input_path}.")
                        self.runlist = [self.runlist]
                    else:
                        alert_dialog("No valid volume or folder given, please give a nift volume, dcm volume, dcm series folder or folder with NifTs.")
            else:
                self.runlist = [self.input_path]
            self.write_to_textbox(f"Runlist: {self.runlist}.\n{len(self.runlist)} volumes detected.\nClick start processing to start.")
        else:
            alert_dialog("No valid volume or folder given, please give a nift volume, dcm volume, dcm series folder or folder with NifTs.")

    def cli_populate_runlist(self):

        if self.input_path is None:
            pass
        elif os.path.exists(self.input_path) and (".nii" in self.input_path or os.path.isdir(self.input_path) or ".dcm" in self.input_path):
            if os.path.isdir(self.input_path):
                print(f"Searching {self.input_path} for files...")
                self.runlist = glob.glob(os.path.join(self.input_path, "*.nii")) + glob.glob(os.path.join(self.input_path, "*.nii.gz"))
                if len(self.runlist) == 0:
                    print("Did not find NifT files. Looking for .dcm series...")
                    self.runlist = glob.glob(os.path.join(self.input_path, "*.dcm"))
                    dcms = [os.path.basename(x) for x in self.runlist]
                    if len(self.runlist) > 0:
                        print(f"Found {dcms} DCM inside folder {self.input_path}.")
                        self.runlist = [self.runlist]
                    else:
                        raise ValueError("No valid volume or folder given, please give a nift volume, dcm volume, dcm series folder or folder with NifTs.")
            else:
                self.runlist = [self.input_path]
            print(f"Runlist: {self.runlist}.\n{len(self.runlist)} volumes detected.\nClick start processing to start.")
        else:
            ValueError("No valid volume or folder given, please give a nift volume, dcm volume, dcm series folder or folder with NifTs.")
    
    def write_to_textbox(self, s):
        if self.cli:
            print(s)
        else:
            self.T.insert(tk.END, f"\n{s}\n")
            self.T.see(tk.END)
        
    def load_file(self):
        self.input_path = file_dialog(dir=False)
        self.populate_runlist()
    
    def load_folder(self):
        alert_dialog("Double click (enter) the directory with the input files.")
        self.input_path = file_dialog(dir=True)
        self.populate_runlist()
    
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if self.pipeline is not None and self.pipeline.is_alive():
                self.write_to_textbox("Closing...")
                self.pipeline.join()
                self.write_to_textbox("Done.")
            self.ws.quit()

    def monitoring_loop(self):
        while True:
            self.monitoring()
            time.sleep(0.1)

    def monitoring(self):
        stats = get_stats()
        for key, value in stats.items():
            getattr(self, key)['value'] = value

    def set_icon(self):
        if not self.cli:
            self.img = ImageTk.PhotoImage(Image.open(ICON_PNG))
            self.canvas.create_image(0, 0, anchor=NW, image=self.img)

    def run(self):
        '''
        Design intent:
            - Plain window, with image/folder loading button and start processing button. 
            - TQDM progress somehow reflected on gui bar
            - Text box with debug output
            - Run in thread
        '''
        if self.cli:
            self.start_processing() 
            self.pipeline.join()
        else:
            self.ws = Tk()
            icon = PhotoImage(file=ICON_PNG)
            self.ws.iconphoto(False, icon)
            self.ws.title(DEFAULT_TITLE)
            self.ws.geometry('1280x720')

            # Canvas
            self.canvas = Canvas(self.ws, width=128, height=128)
            self.canvas.pack(side='top')
            self.set_icon()

            # Text output
            scroll = Scrollbar(self.ws)
            self.T = Text(self.ws, height=20, width=60, font=("Sans", 14), yscrollcommand=scroll.set)        
            scroll.config(command=self.T.yview)
            scroll.pack(side='right', fill='y')
            self.T.pack(side='top', fill='both')

            
            self.write_to_textbox(f"Welcome to MEDSeg! {DEFAULT_TITLE}")
            if self.args.output_folder is not None:
                os.makedirs(self.args.output_folder, exist_ok=True)
                self.write_to_textbox(f"Results will be in the '{self.args.output_folder}' folder")
            if self.args.cpu:
                self.write_to_textbox(f"Forcing CPU usage. Prediction might take a while.")
            
            general_progress = Label(self.ws, text="General Progress")
            general_progress.pack(side='bottom')
            self.general_progress = Progressbar(self.ws, orient=HORIZONTAL, length=600, mode='determinate')
            self.general_progress.pack(side='bottom', fill='x')
            iter_progress = Label(self.ws, text="Processing Progress")
            iter_progress.pack(side='bottom')
            self.iter_progress = Progressbar(self.ws, orient=HORIZONTAL, length=600, mode='determinate')
            self.iter_progress.pack(side='bottom', fill='x')

            # Monitoring bars
            cpu_label = Label(self.ws, text="CPU")
            cpu_label.pack(side='left')
            self.cpu = Progressbar(self.ws, orient=VERTICAL, length=60, mode='determinate')
            self.cpu.pack(side='left')
            
            gpu_label = Label(self.ws, text="GPU")
            gpu_label.pack(side='left')
            self.gpu = Progressbar(self.ws, orient=VERTICAL, length=60, mode='determinate')
            self.gpu.pack(side='left')
            
            ram_label = Label(self.ws, text="RAM")
            ram_label.pack(side='left')
            self.cpu_ram = Progressbar(self.ws, orient=VERTICAL, length=60, mode='determinate')
            self.cpu_ram.pack(side='left')
            
            vram_label = Label(self.ws, text="VRAM")
            vram_label.pack(side='left')
            self.gpu_ram = Progressbar(self.ws, orient=VERTICAL, length=60, mode='determinate')
            self.gpu_ram.pack(side='left')

            self.display = tk.IntVar(value=1)
            c1 = tk.Checkbutton(self.ws, text='Display result', variable=self.display, onvalue=1, offvalue=0, state='active')
            c1.config(font=("Sans", "14"))
            c1.pack(side='left')

            self.long = tk.IntVar(value=0)
            c2 = tk.Checkbutton(self.ws, text='Long prediction', variable=self.long, onvalue=1, offvalue=0, state='active')
            c2.config(font=("Sans", "14"))
            c2.pack(side='left')

            boldStyle = Style ()
            boldStyle.configure("Bold.TButton", font = ('Sans','10','bold'))
            Button(self.ws, text='Start processing', command=self.start_processing, style="Bold.TButton").pack(side='right', ipady=10, pady=10, ipadx=5, padx=5)        
            Button(self.ws, text='Load image ', command=self.load_file).pack(side='right', ipady=10, pady=10, ipadx=5, padx=5)
            Button(self.ws, text='Load folder', command=self.load_folder).pack(side='right', ipady=10, pady=10, ipadx=5, padx=5)
            
            self.ws.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            monitoring = threading.Thread(target=self.monitoring_loop)
            monitoring.daemon = True
            monitoring.start()
            self.ws.mainloop()
        


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def test_worker(info_q):
    for info in range(10):
        info_q.put(info)
        time.sleep(1)

    info_q.put(None)
    