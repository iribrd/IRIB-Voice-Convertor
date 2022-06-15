import os
import pickle
import librosa
import torch
import numpy as np
from math import ceil

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass
			
for item in read_from_pickle('test2000.pkl'):
        print(repr(item))
        
