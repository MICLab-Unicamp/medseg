import argparse
import multiprocessing as mp

from gui import MainWindow   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cache_folder', type=str, default="cache")
    parser.add_argument('-o', '--output_folder', type=str, default="output")
    parser.add_argument('-bs', '--batch_size', type=int, default=1)
    parser.add_argument('-m', '--model_path', type=str, default="weights/best_coedet.ckpt")
    args = parser.parse_args()

    MainWindow(args, mp.Queue()).join()
    
    