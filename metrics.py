# BSD License
#
# Copyright (c) 2016-present, Miguel González-Fierro. All rights reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name Miguel González-Fierro nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, log_loss, recall_score, f1_score


def evaluate_metrics(y_true, y_pred, metrics):
    res = {}
    for metric_name, metric in metrics.items():
        res[metric_name] = metric(y_true, y_pred)
    return res


def classification_metrics_binary_prob(y_true, y_prob, threshold=0.5):
    y_pred = np.where(y_prob > threshold, 1, 0)
    metrics = {
        "Accuracy":  accuracy_score,
        "Precision": precision_score,
        "Recall":    recall_score,
        "Log_Loss":  lambda real, pred: log_loss(real, y_prob, eps=1e-5),
        # yes, I'm using y_prob here!
        "AUC":       lambda real, pred: roc_auc_score(real, y_prob),
        "F1":        f1_score,
    }
    return evaluate_metrics(y_true, y_pred, metrics)


def classification_metrics_multilabel(y_true, y_pred, labels):
    metrics = {
        "Accuracy":  accuracy_score,
        "Precision": lambda real, pred: precision_score(real, pred, labels,
                                                        average="weighted"),
        "Recall":    lambda real, pred: recall_score(real, pred, labels,
                                                     average="weighted"),
        "F1":        lambda real, pred: f1_score(real, pred, labels,
                                                 average="weighted"),
    }
    return evaluate_metrics(y_true, y_pred, metrics)


def classification_metrics_average(y_true, y_pred, avg):
    metrics = {
        "Accuracy":  accuracy_score,
        "Precision": lambda real, pred: precision_score(real, pred, average=avg),
        "Recall":    lambda real, pred: recall_score(real, pred, average=avg),
        "F1":        lambda real, pred: f1_score(real, pred, average=avg),
    }
    return evaluate_metrics(y_true, y_pred, metrics)


def classification_metrics(y_true, y_pred):
    metrics = {
        "Accuracy":  accuracy_score,
        "Precision": precision_score,
        "Recall":    recall_score,
        "AUC":       roc_auc_score,
        "F1":        f1_score,
    }
    return evaluate_metrics(y_true, y_pred, metrics)
