#!/usr/bin/env python
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import argparse
import json
import ast
import psutil
import algorithms
from metrics import get_metrics
from datasets import prepare_dataset
import gc

def print_sys_info():
    try:
        import xgboost  # pylint: disable=import-outside-toplevel
        print("Xgboost : %s" % xgboost.__version__)
    except ImportError:
        pass
    try:
        import lightgbm  # pylint: disable=import-outside-toplevel
        print("LightGBM: %s" % lightgbm.__version__)
    except (ImportError, OSError):
        pass
    try:
        import catboost  # pylint: disable=import-outside-toplevel
        print("Catboost: %s" % catboost.__version__)
    except ImportError:
        pass
    print("System  : %s" % sys.version)
    print("#CPUs   : %d" % psutil.cpu_count(logical=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark xgboost/lightgbm/catboost on real datasets")
    parser.add_argument("-root", default="/opt/gbm-datasets",
                        type=str, help="The root datasets folder")
    parser.add_argument("-input", required=True, help='JSON file that contains experiment parameters')
    parser.add_argument("-output", default=sys.path[0] + "/results.json", type=str,
                        help="Output json file with runtime/accuracy stats")
    parser.add_argument("-verbose", action="store_true", help="Produce verbose output")

    args = parser.parse_args()

    return args


# benchmarks a single dataset
def benchmark(algo, dataset_dir, dataset_parameters, algorithm_parameters):
    data = prepare_dataset(dataset_dir, dataset_parameters['dataset_name'],
                           dataset_parameters['nrows'])
    results = {}
    runner = algorithms.Algorithm.create(algo)
    with runner:
        train_time = runner.fit(data, algorithm_parameters)
        pred = runner.test(data)
        result = {
                   "train_time" : train_time,
                   "accuracy": get_metrics(data, pred)
                 }
    del data
    gc.collect()
    return result

def main():
    args = parse_args()
    print_sys_info()

    with open(args.input) as fp:
        experiments = json.load(fp)
        results = []
        for exp in experiments['experiments']:
                output = exp.copy()
                dataset_parameters = exp['dataset_parameters']
                if not 'nrows' in dataset_parameters.keys():
                    dataset_parameters['nrows'] = None
                algorithm_parameters = exp['algorithm_parameters']
                dataset_dir = os.path.join(
                    args.root, dataset_parameters['dataset_name'])
                res = benchmark(exp['algo'], dataset_dir, dataset_parameters,
                                algorithm_parameters)
                output.update({'result' : res})
                results.append(output)
    # print(json.dumps({ 'experiments' : results }, indent = 2, sort_keys = True))
    results_str = json.dumps({ 'experiments' : results }, indent=2, sort_keys=True)

    with  open(args.output, "w") as fp:
        fp.write(results_str + "\n")

    print("Results written to file '%s'" % args.output)


if __name__ == "__main__":
    main()
