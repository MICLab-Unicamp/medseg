import argparse
import multiprocessing as mp

from coedet.gui import MainWindow   


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_folder', type=str, default=None, help="Use to indicate output folder without using the GUI. Will try to create if it doesn't exist.")
    parser.add_argument('--cpu', action="store_true", help="Forces using only the CPU.")
    parser.add_argument('-bs', '--batch_size', type=int, default=1)
    parser.add_argument('-m', '--model_path', type=str, default="best_coedet.ckpt")
    parser.add_argument('-win_itk_path', type=str, default="C:\\Program Files\\ITK-SNAP 3.8\\bin\\ITK-SNAP.exe", help="Path for ITKSnap exe location in Windows. Default: 'C:\\Program Files\\ITK-SNAP 3.8\\bin\\ITK-SNAP.exe'")
    parser.add_argument('-linux_itk_path', type=str, default="itksnap", help="Command for itksnap execution in Linux. Default: 'itksnap'")
    args = parser.parse_args()

    MainWindow(args, mp.Queue()).join()


if __name__ == "__main__":
    main()
    