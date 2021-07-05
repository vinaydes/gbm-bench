from enum import Enum
from urllib.request import urlretrieve
import tqdm
import numpy as np
import os

pbar = None

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = tqdm.tqdm(total=total_size / 1024, unit='kB')

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(block_size / 1024)
    else:
        pbar.close()
        pbar = None


def retrieve(url, filename=None):
    return urlretrieve(url, filename, reporthook=show_progress)

class LearningTask(Enum):
    REGRESSION = 1
    CLASSIFICATION = 2
    MULTICLASS_CLASSIFICATION = 3


class Data:  # pylint: disable=too-few-public-methods,too-many-arguments
    def __init__(self, X_train, X_test, y_train, y_test, learning_task, qid_train=None,
                 qid_test=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.learning_task = learning_task
        # For ranking task
        self.qid_train = qid_train
        self.qid_test = qid_test

def npy_to_data(path, nrows=None, learning_task=LearningTask.REGRESSION):
    suffix = f'.npy' if nrows is None else f'-{nrows}.npy'
    if os.path.isfile(os.path.join(path, f'X_train{suffix}')): # no need to check for each file as they are grouped
        X_train = np.load(os.path.join(path, f'X_train{suffix}'))
        X_test = np.load(os.path.join(path, f'X_test{suffix}'))
        y_train = np.load(os.path.join(path, f'y_train{suffix}'))
        y_test = np.load(os.path.join(path, f'y_test{suffix}'))

        return Data(X_train, X_test, y_train, y_test, learning_task)
    else:
        return None

def data_to_npy(path, data, nrows=None):
    if(not os.path.isdir(path)):
        os.mkdir(path)
    suffix = f'.npy' if nrows is None else f'-{nrows}.npy'
    X_train = np.save(os.path.join(path, f'X_train{suffix}'), data.X_train, allow_pickle=False)
    X_test = np.save(os.path.join(path, f'X_test{suffix}'), data.X_test, allow_pickle=False)
    y_train = np.save(os.path.join(path, f'y_train{suffix}'), data.y_train, allow_pickle=False)
    y_test = np.save(os.path.join(path, f'y_test{suffix}'), data.y_test, allow_pickle=False)
