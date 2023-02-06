#!/usr/bin/env python
"""---- author: Moritz Baumgarten ---- 
Provides a class for statistical tools such as the blocking 
analysis for the shift.
"""

import numpy as np
import scipy.linalg as la
import sys, os, shutil
from typing import Tuple, Sequence

####CUSTOM IMPORTS
from pyblock import blocking


class Statistics():
        r"""Class to perform statistical analysis of energy 
        estimators (and populations later)"""
        def __init__(self, Ss: np.ndarray, params: dict) -> None:
                self.params = params
                self.Ss = Ss
                self.i = -1 + int(
                    np.log2(self.params['it_nr'] - self.params['delay'])
                )
                self.S = np.array(Ss[self.params['it_nr'] - 2**self.i + 1 : ])
                self.n = len(self.S)

        def analyse(self, data: np.ndarray = None):
                if data is None: data = self.S
                data_summary = blocking.reblock(data)
                err_bound = np.max([
                    data_summary[i][4] for i in range(len(data_summary))
                ])
                np.save(
                    os.path.join(self.params['workdir'], 'std_err.npy'), 
                    [data_summary[i][4] for i in range(len(data_summary))]
                )
                block = blocking.find_optimal_block(len(data), data_summary)
                if np.isnan(block[0]):
                        return 0
                else:
                        error = self.binning(block[0])
                        print('Error by binning:        ', error, block)
                        return error

      #IMPLEMENT BINNING HERE TODO, calculcate correlation length to get n_bins
        def binning(self, n_bins: int) -> float:                
                m_bins = int(len(self.S) / n_bins)
                data = np.array([
                    self.S[m_bins*i : m_bins*(i+1)] for i in range(n_bins)
                ])
                data /= m_bins
                mean = np.mean(data, axis = 0)
                error = (1 / n_bins / (n_bins-1)) * np.sum( (data-mean)**2 )
                return error

if __name__ == '__main__':

        Ss = np.random.random(2000)
        params = {'it_nr': 1990, 'delay': 100, 'workdir': '.'}

        stats = Statistics(Ss = Ss, params = params)
        error = stats.analyse()
        print(error)



