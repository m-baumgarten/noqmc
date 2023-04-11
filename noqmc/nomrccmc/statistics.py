#!/usr/bin/env python
"""---- author: Moritz Baumgarten ---- 
Provides a class for statistical tools such as the blocking 
analysis for the shift.
"""

import numpy as np
import collections
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
                    np.log2(self.params.it_nr - self.params.delay)
                ) -1
                self.S = np.array(Ss[self.params.it_nr - 2**self.i + 1 : ])
                self.n = len(self.S)

        def analyse(self, data: np.ndarray = None) -> int:
                if data is None: data = self.S
                self.data_summary = blocking.reblock(data)
                self.block = blocking.find_optimal_block(
                        len(data), self.data_summary
                )[0]
                if np.isnan(self.block):
                        print('Blocking analysis did not converge.')
                        self.block = -1
                #print('Optimal Block:   ', self.block)
                self.mean = self.data_summary[self.block].mean
                print('Data Summary:    ', self.mean)
                return self.block

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
        print(stats.data_summary[0][1])
        print(error)



