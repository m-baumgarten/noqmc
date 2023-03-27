#!/usr/bin/env python
"""---- author: Moritz Baumgarten ----# 
Implementation of a nonorthogonal multireference configuration
interaction quantum Monte Carlo (NOMRCI-QMC) method.
The Hilbert space for a common NOMRCI-QMC calculation is generated
by a subset of possible exitations of the reference determinats
(generally obtained from SCF metadynamics)
Based on Booth, Thom and Alavi [2009], and Thom and Head-Gordon [2008]
"""

import numpy as np
import sys
from typing import (
    Tuple, 
    Sequence,
)    

####QCMAGIC IMPORTS
from qcmagic.core.cspace.basis.basisset import ConvolvedBasisSet
from qcmagic.core.backends.nonorthogonal_backend import (
    calc_overlap, 
    calc_hamiltonian, 
)

####CUSTOM IMPORTS
from noqmc.nomrccmc.system import System
from noqmc.nomrccmc.propagator import calc_mat_elem

class Propagator(System):
        r"""
        Class for propagation of the wavefunction/walkers in imaginary 
        time. Inherits determinant methods from the System class. 
        """

        def __init__(self, system: System) -> None:
                r"""Inherits parameters from System object.

		:param system: System object containing 
                               information about Hilbert space
                """
                self.__dict__.update(system.__dict__)
                
                #Initialize Properties of the Propagator 
                self.Nws = np.empty(self.params['it_nr'], dtype = int)
                self.n = []
                self.curr_it = 0
                self.E_ref = self.E_HF

                self.coeffs = np.zeros([self.params['it_nr']+1, self.params['dim']], dtype = int)
                self.coeffs[0,:] = self.initial.copy()

                self.E_proj = np.empty(self.params['it_nr']+1)
                self.E_proj[0] = self.E_NOCI - self.E_ref

                self.Ss = np.empty(self.params['it_nr']+1)
                self.S = self.Ss[0] = 0

        def E(self) -> None:
                r"""Calculates energy estimator at current iteration
                based on the cached Hamiltonian and overlap matrix."""
                
                coeffs = self.coeffs[self.curr_it,:]
                overlap_tmp = np.nan_to_num(self.overlap)
                H_tmp = np.nan_to_num(self.H)
                
                #Get Index of Maximum Value
                self.index = np.where(
                        np.abs(coeffs) == np.max(np.abs(coeffs))
                )[0][0] 
                
                E_proj = np.einsum('i,i->', H_tmp[self.index,:], coeffs)
                E_proj /= np.einsum('i,i->', overlap_tmp[self.index, :], coeffs)
                self.E_proj[self.curr_it+1] = E_proj

        def Shift(self) -> None:
                r"""Updates shift every A-th iteration.

                :param A: Interval of reevaluation of S
		:param c: Empirical daming parameter c
		"""
                N_new = self.Nws[self.curr_it]
                N_old = self.Nws[self.curr_it - self.params['A']]
                self.n.append(N_new/N_old)
                self.S -= self.params['c'] / (self.params['A'] * self.params['dt']) * np.log(N_new / N_old)

        def Nw(self) -> None:
                r"""Updates total number of walkers resident outside 
                the kernel of S.
                """
                overlap_tmp = np.nan_to_num(self.overlap) 
                proj = np.einsum('ij,j->i', overlap_tmp, self.coeffs[self.curr_it, :])
                self.Nws[self.curr_it] = np.linalg.norm(proj, ord=1)

        def population_dynamics(self) -> None:
                r"""Spawning/Death in one step due to nonorthogonality. 
		Writes changes to current wavefunction to sp_coeffs.
                """
                sp_coeffs = np.zeros(self.params['dim'], dtype=int)

                for i,coeff in enumerate(self.coeffs[self.curr_it, :]):
                        
                        sign_coeff = np.sign(coeff)
                        
                        #Uniform Excitation Generation -> TODO Generalize
                        js = np.random.randint(0, self.params['dim'], size=abs(coeff))

                        #Prepare self.H and self.overlap:
                        key_i = self.index_map_rev[i]
                        det_i = self.generate_det(key_i)
                        occ_i = det_i.occupied_coefficients
                        
                        set_js, index, counts = np.unique(
                            js, return_index=True, return_counts=True
                        )
                        #TODO generate array of bools here indicating whether excitation to j is allowed from i
                        #and zip in in loop over set_js
                        ####---LOCALIZATION STEP
                        #for j in set_js:
                        #        key_j = self.index_map_rev[j]
                                
                        ####---END


                        for j in set_js:
                                MAT_ELEM_CACHED = not np.isnan(self.H[i,j])
                                if MAT_ELEM_CACHED:
                                        continue
                                det_j = self.get_det(j)
                                occ_j = det_j.occupied_coefficients
                                
                                elem = calc_mat_elem(
                                    occ_i=occ_i, occ_j=occ_j, 
                                    cbs=self.cbs, enuc=self.enuc, 
                                    sao=self.sao, hcore=self.hcore,
                                    E_ref=self.E_ref
                                )
                                self.H[i,j], self.H[j,i] = elem[0], elem[1]
                                self.overlap[i,j], self.overlap[j,i] = elem[2], elem[3]

                        spawning_probs = [
                            self.params['dim'] * self.params['dt'] 
                            * (self.H[i,j] - (self.S) * self.overlap[i,j])
                            for j in set_js
                        ]
                        rand_vars = np.random.random(size=(len(spawning_probs)))
                        for j, n, ps, r in zip(set_js,counts,spawning_probs,rand_vars):
                                ps_s = n * ps
                                s_int = int(ps_s)
                                b = ps_s - s_int
                                s_int += (r < np.abs(b)) * np.sign(b)
                                sp_coeffs[j] -= sign_coeff * s_int

                #Annihilation
                self.coeffs[self.curr_it+1, :] = sp_coeffs
                self.coeffs[self.curr_it+1, :] += self.coeffs[self.curr_it, :]

                if self.params['verbosity']:
                        print(f'{self.curr_it} new spawns:      ',
                            np.linalg.norm(self.coeffs[self.curr_it+1, :], ord = 1), 
                            self.S 
                        )

        def run(self) -> None:
                r"""Executes the FCIQMC algorithm.
                """
                
                for i in range(self.params['it_nr']):

                        #Only measure Number of Walkers outside of ker(S)
                        self.Nw()
                        
                        SHIFT_UPDATE = i%self.params['A'] == 0 and i > self.params['delay']
                        if SHIFT_UPDATE: 
                                self.Shift()
                        self.Ss[self.curr_it+1] = self.S
                        
                        self.population_dynamics()
                        
                        self.E()
                        self.curr_it += 1


