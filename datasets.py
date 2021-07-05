# MIT License
#
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file, fetch_covtype
import pandas as pd


from dataset_utils import retrieve, LearningTask, Data, npy_to_data, data_to_npy

def __prepare_airline(dataset_folder, dataset_parameters, regression=False):  # pylint: disable=too-many-locals
    nrows = dataset_parameters['nrows']
    url = 'http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2'
    if regression:
        task = LearningTask.REGRESSION
    else:
        task = LearningTask.CLASSIFICATION
    local_url = os.path.join(dataset_folder, os.path.basename(url))
    npy_path = os.path.join(dataset_folder, 'npy')

    # load if npy arrays if exists
    if os.path.isdir(npy_path):
        ret_data = npy_to_data(npy_path, nrows=nrows, learning_task=task)
        if ret_data is not None:
            return ret_data

    if not os.path.isfile(local_url):
        retrieve(url, local_url)

    cols = [
        "Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime",
        "CRSArrTime", "UniqueCarrier", "FlightNum", "ActualElapsedTime",
        "Origin", "Dest", "Distance", "Diverted", "ArrDelay"
    ]

    # load the data as int16
    dtype = np.int16

    dtype_columns = {
        "Year": dtype, "Month": dtype, "DayofMonth": dtype, "DayofWeek": dtype,
        "CRSDepTime": dtype, "CRSArrTime": dtype, "FlightNum": dtype,
        "ActualElapsedTime": dtype, "Distance":
            dtype,
        "Diverted": dtype, "ArrDelay": dtype,
    }

    df = pd.read_csv(local_url,
                     names=cols, dtype=dtype_columns, nrows=nrows)

    # Encode categoricals as numeric
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].astype("category").cat.codes

    # Turn into binary classification problem
    if not regression:
        df["ArrDelay"] = 1 * (df["ArrDelay"] > 0)

    X = df[df.columns.difference(["ArrDelay"])].to_numpy(dtype=np.float32)
    y = df["ArrDelay"].to_numpy(dtype=np.float32)
    del df
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test, task)

    # store data as npy arrrays
    data_to_npy(npy_path, data, nrows=nrows)

    return data

def prepare_airline(dataset_folder, dataset_parameters):
    return __prepare_airline(dataset_folder, dataset_parameters, False)

def prepare_airline_regression(dataset_folder, dataset_parameters):
    return __prepare_airline(dataset_folder, dataset_parameters, True)

def prepare_bosch(dataset_folder, dataset_parameters):
    nrows = dataset_parameters['nrows']
    filename = "train_numeric.csv.zip"
    local_url = os.path.join(dataset_folder, filename)
    npy_path = os.path.join(dataset_folder, 'npy')

    # load if npy arrays if exists
    if os.path.isdir(npy_path):
        ret_data = npy_to_data(npy_path, nrows=nrows, learning_task=LearningTask.CLASSIFICATION)
        if ret_data is not None:
            return ret_data

    os.system("kaggle competitions download -c bosch-production-line-performance -f " +
              filename + " -p " + dataset_folder)
    X = pd.read_csv(local_url, index_col=0, compression='zip', dtype=np.float32,
                    nrows=nrows)
    y = X.iloc[:, -1].to_numpy(dtype=np.float32)
    X.drop(X.columns[-1], axis=1, inplace=True)
    X = X.to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test, LearningTask.CLASSIFICATION)

    # store data as npy arrrays
    data_to_npy(npy_path, data, nrows=nrows)

    return data

def prepare_fraud(dataset_folder, dataset_parameters):
    nrows = dataset_parameters['nrows']
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    filename = "creditcard.csv"
    local_url = os.path.join(dataset_folder, filename)
    npy_path = os.path.join(dataset_folder, 'npy')

    # load if npy arrays if exists
    if os.path.isdir(npy_path):
        ret_data = npy_to_data(npy_path, nrows=nrows, learning_task=LearningTask.CLASSIFICATION)
        if ret_data is not None:
            return ret_data

    os.system("kaggle datasets download mlg-ulb/creditcardfraud -f" +
              filename + " -p " + dataset_folder)
    df = pd.read_csv(local_url + ".zip", dtype=np.float32, nrows=nrows)
    X = df[[col for col in df.columns if col.startswith('V')]].to_numpy(dtype=np.float32)
    y = df['Class'].to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test, LearningTask.CLASSIFICATION)

    # store data as npy arrrays
    data_to_npy(npy_path, data, nrows=nrows)

    return data

def prepare_higgs(dataset_folder, dataset_parameters):
    nrows = dataset_parameters['nrows']
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    local_url = os.path.join(dataset_folder, os.path.basename(url))

    npy_path = os.path.join(dataset_folder, 'npy')

    # load if npy arrays if exists
    if os.path.isdir(npy_path):
        ret_data = npy_to_data(npy_path, nrows=nrows, learning_task=LearningTask.CLASSIFICATION)
        if ret_data is not None:
            return ret_data

    if not os.path.isfile(local_url):
        retrieve(url, local_url)
    higgs = pd.read_csv(local_url, nrows=nrows)
    X = higgs.iloc[:, 1:].to_numpy(dtype=np.float32)
    y = higgs.iloc[:, 0].to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    data = Data(X_train, X_test, y_train, y_test, LearningTask.CLASSIFICATION)

    # store data as npy arrrays
    data_to_npy(npy_path, data, nrows=nrows)

    return data

def prepare_year(dataset_folder, dataset_parameters):
    nrows = dataset_parameters['nrows']
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt' \
          '.zip'
    local_url = os.path.join(dataset_folder, os.path.basename(url))
    npy_path = os.path.join(dataset_folder, "npy")

    if os.path.isdir(npy_path):
        ret_data = npy_to_data(npy_path, nrows=nrows, learning_task=LearningTask.REGRESSION)
        if ret_data is not None:
            return ret_data

    if not os.path.isfile(local_url):
        retrieve(url, local_url)
    year = pd.read_csv(local_url, nrows=nrows, header=None)
    X = year.iloc[:, 1:].to_numpy(dtype=np.float32)
    y = year.iloc[:, 0].to_numpy(dtype=np.float32)

    if nrows is None:
        # this dataset requires a specific train/test split,
        # with the specified number of rows at the start belonging to the train set,
        # and the rest being the test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False,
                                                            train_size=463715,
                                                            test_size=51630)
    else:
        print(
            "Warning: nrows is specified, not using predefined test/train split for "
            "YearPredictionMSD.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                            test_size=0.2,
                                                            )

    data = Data(X_train, X_test, y_train, y_test, LearningTask.REGRESSION)

    # store data as npy arrrays
    data_to_npy(npy_path, data, nrows=nrows)
    return data

def prepare_epsilon(dataset_folder, dataset_parameters):
    nrows = dataset_parameters['nrows']
    url_train = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary' \
                '/epsilon_normalized.bz2'
    url_test = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary' \
               '/epsilon_normalized.t.bz2'
    local_url_train = os.path.join(dataset_folder, os.path.basename(url_train))
    local_url_test = os.path.join(dataset_folder, os.path.basename(url_test))

    npy_path = os.path.join(dataset_folder, 'npy')

    # load if npy arrays if exists
    if os.path.isdir(npy_path):
        ret_data = npy_to_data(npy_path, nrows=nrows, learning_task=LearningTask.CLASSIFICATION)
        if ret_data is not None:
            return ret_data

    if not os.path.isfile(local_url_train):
        retrieve(url_train, local_url_train)
    if not os.path.isfile(local_url_test):
        retrieve(url_test, local_url_test)

    X_train, y_train = load_svmlight_file(local_url_train,
                                          dtype=np.float32)
    X_test, y_test = load_svmlight_file(local_url_test,
                                        dtype=np.float32)
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    y_train[y_train <= 0] = 0
    y_test[y_test <= 0] = 0

    if nrows is not None:
        print("Warning: nrows is specified, not using predefined test/train split for epsilon.")

        X_train = np.vstack((X_train, X_test))
        y_train = np.append(y_train, y_test)
        X_train = X_train[:nrows]
        y_train = y_train[:nrows]
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=77,
                                                            test_size=0.2,
                                                            )

    data = Data(X_train, X_test, y_train, y_test, LearningTask.CLASSIFICATION)

    # store data as npy arrrays
    data_to_npy(npy_path, data, nrows=nrows)

    return data

def prepare_covtype(dataset_folder, dataset_parameters):  # pylint: disable=unused-argument
    nrows = dataset_parameters['nrows']
    X, y = fetch_covtype(return_X_y=True)  # pylint: disable=unexpected-keyword-arg
    if nrows is not None:
        X = X[0:nrows]
        y = y[0:nrows]
    # Labele range in covtype start from 1, making it start from 0
    y = y - 1

    X = np.float32(X)
    y = np.int32(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=77,
                                                        test_size=0.2,
                                                        )
    return Data(X_train, X_test, y_train, y_test, LearningTask.MULTICLASS_CLASSIFICATION)
