#!/usr/bin/env python
#---- author: Moritz Baumgarten ----# 
#Implementation of a nonorthogonal multireference configuration
#interaction quantum Monte Carlo (NOMRCI-QMC) method.
#The Hilbert space for a common NOMRCI-QMC calculation is generated
#by a subset of possible exitations of the reference determinats
#(generally obtained from SCF metadynamics)
#Based on Booth, Thom and Alavi [2009], and Thom and Head-Gordon [2008]

import numpy as np
import multiprocessing
import sys, os
from typing import (
    Tuple, 
    Sequence,
)    

####QCMAGIC IMPORTS
from qcmagic.core.backends.nonorthogonal_backend import (
    calc_overlap, 
    calc_hamiltonian, 
)
from qcmagic.core.sspace.single_determinant import SingleDeterminant

####CUSTOM IMPORTS
from noqmc.nomrccmc.system import (
    System,
    calc_mat_elem,
)

norm = lambda x: x / np.linalg.norm(x, ord = 2) 

def mute():
    sys.stdout = open(os.devnull, 'w')

class Propagator(System):
        r"""Class for propagation of the wavefunction/walkers in imaginary time."""

        def __init__(self, system: System) -> None:
                r"""Inherits parameters from System object.

		:param system: System object containing information about Hilbert space"""
                self.__dict__.update(system.__dict__)
                self.E_ref = self.E_HF
                
                self.E_proj = np.empty(self.params['it_nr']+1)
                self.E_proj[0] = self.E_NOCI - self.E_ref        

                self.Ss = np.empty(self.params['it_nr']+1)
                self.Nws = np.empty(self.params['it_nr'], dtype = int)
                self.coeffs = np.zeros(
                    [self.params['it_nr']+1, self.params['dim']]
                )
                self.coeffs[0,:] = norm(self.initial.copy())
                
                self.coeffs[0,:] = np.random.random(722)
                self.coeffs[0,:] /= np.linalg.norm(self.coeffs[0,:])
                
                self.S = self.Ss[0] = 0
                self.curr_it = 0
                self.n = []

        def calculate_operators(self):
                r"""Calculates the Hamiltonian and overlap"""        
                pool = multiprocessing.Pool(
                    processes = multiprocessing.cpu_count(), 
                    initializer = mute
                )
                processes = {}

                isnan = np.isnan(self.H)
                indices = np.where(isnan)
                
                for i,j in zip(indices[0], indices[1]):
                        det_i = self.get_det(i)
                        det_j = self.get_det(j)
                        occ_i = det_i.occupied_coefficients
                        occ_j = det_j.occupied_coefficients

                        processes[(i,j)] = pool.apply_async(
                            calc_mat_elem, 
                            [occ_i, occ_j, self.cbs, self.enuc, self.sao, self.hcore, self.E_ref]
                        )

                pool.close()
                pool.join()
                for i,j in zip(indices[0], indices[1]):
                        processes[(i,j)] = processes[(i,j)].get()
                        self.H[i,j] = processes[(i,j)][0]
                        self.H[j,i] = processes[(i,j)][1]
                        self.overlap[i,j] = processes[(i,j)][2]
                        self.overlap[j,i] = processes[(i,j)][3]

                self.propagator = np.eye(self.params['dim']) - self.params['dt'] * self.H

        def E(self) -> None:
                r"""Calculates energy estimator at current iteration."""
                
                coeffs = self.coeffs[self.curr_it,:]
                index = np.where(
                    np.abs(coeffs) == np.max(np.abs(coeffs))
                )[0][0] #get index of maximum value 
                E_proj = np.einsum('i,i->', self.H[index,:], coeffs)
                E_proj /= np.einsum('i,i->', self.overlap[index, :], coeffs)
                self.E_proj[self.curr_it+1] = E_proj
                return E_proj

        def population_dynamics(self) -> None:    #parallelize for each scf solution
                r"""Spawning/Death in one step due to nonorthogonality. 
		Writes changes to current wavefunction to self.sp_coeffs."""

                E_proj = self.E()

                new = np.dot(self.propagator, self.coeffs[self.curr_it, :])
                new = norm(new)
                self.coeffs[self.curr_it+1, :] = new

                print(f'{self.curr_it} E_proj:      ', 
                    self.E_proj[self.curr_it] 
                )

        def run(self) -> None:
                r"""Executes what is essentially the power method.
                """
                self.calculate_operators()                
                for i in range(self.params['it_nr']):
                        self.curr_it = i
                        self.Nws[self.curr_it] = sum([int(np.round(np.abs(c))) for c in self.coeffs[self.curr_it, :]])
                        self.population_dynamics()
                print(np.linalg.eigh(self.overlap))

if __name__ == '__main__':      
        pass



