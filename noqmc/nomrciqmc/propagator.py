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
#import scipy.linalg as la
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


class Propagator(System):
        r"""Class for propagation of the wavefunction/walkers in imaginary time."""

        def __init__(self, system: System) -> None:
                r"""Inherits parameters from System object.

		:param system: System object containing information about Hilbert space"""
                self.__dict__.update(system.__dict__)
                self.E_proj = np.empty(self.params['it_nr'])
                self.Ss = np.empty(self.params['it_nr']+1)
                self.Nws = np.empty(self.params['it_nr'], dtype = int)
                self.coeffs = np.zeros([self.params['it_nr']+1, self.params['dim']], dtype = int)
                self.coeffs[0,:] = self.initial.copy()
                self.S = self.Ss[0] = 0
                self.curr_it = 0
                self.n = []

        def E(self) -> None:
                r"""Calculates energy estimator at current iteration."""
                
                coeffs = self.coeffs[self.curr_it,:]
                overlap_tmp = np.nan_to_num(self.overlap)
                H_tmp = np.nan_to_num(self.H)
                self.index = np.where(np.abs(coeffs) == np.max(np.abs(coeffs)))[0][0] #get index of maximum value 
                E_proj = np.einsum('i,i->', H_tmp[self.index,:], coeffs)
                E_proj /= np.einsum('i,i->', overlap_tmp[self.index, :], coeffs)
                self.E_proj[self.curr_it] = E_proj

        def reeval_S(self, A: float = None, c: float = 0.001) -> None:
                r"""Updates shift every A-th iteration.

                :param A:
		:param c:
		:param dt:"""
                N_new = self.Nws[self.curr_it]
                N_old = self.Nws[self.curr_it - self.params['A']]
                self.n.append(N_new/N_old)
                self.S -= c / ( self.params['A'] * self.params['dt'] ) * np.log( N_new / N_old )


        def excitation_generation(self, det_key) -> str:
                r"""Decision tree for generating an excitation, given that we start on a
                determinant with ex_str = key1. Check whether generated excitation is in 
                Hilbert space. Would be best to choose a key from index_map 
                
                :param key1: Specifies reference for excitation
                
                :returns: ex_str of excitation to be generated"""
                
                #FIRST:  choose nr_scf (maybe based on noci overlap?)
                r = np.random.randint(low = 0, high = self.params['dim'])
                return self.index_map_rev[r]
                
        def population_dynamics(self) -> None:    #parallelize for each scf solution
                r"""Spawning/Death in one step due to nonorthogonality. 
		Writes changes to current wavefunction to self.sp_coeffs."""
                self.sp_coeffs = np.zeros(self.params['dim'], dtype = int)
                s_int = 1
                #could store spawning probs in a matrix

                for i,coeff in enumerate(self.coeffs[self.curr_it, :]):
                        
                        sign_coeff = np.sign(coeff)
                        key_i = self.index_map_rev[i]
                        js = np.random.randint(0,self.params['dim'],size=abs(coeff))

                        #prepare self.H and self.overlap:
                        det_i = self.generate_det(key_i)
                        occ_i = det_i.occupied_coefficients
                        set_js, index, counts = np.unique(
                            js, return_index = True, return_counts = True
                        )

                        for j in set_js:
                                if not np.isnan(self.H[i,j]):
                                        continue
                                det_j = self.get_det(j)
                                occ_j = det_j.occupied_coefficients
                                self.H[i,j], self.H[j,i] = calc_hamiltonian(
                                    cws = occ_i, cxs = occ_j, cbs = self.cbs, 
                                    enuc = self.enuc, holo = False, _sao = self.sao, 
                                    _hcore = self.hcore
                                )
                                self.overlap[i,j], self.overlap[j,i] = calc_overlap(
                                    cws = occ_i, cxs = occ_j, cbs = self.cbs, 
                                    holo = False, _sao = self.sao
                                )
                                self.H[i,j] -= self.E_NOCI * self.overlap[i,j]
                                if i != j:
                                        self.H[j,i] -= self.E_NOCI * self.overlap[j,i]
                        
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
                                self.sp_coeffs[j] -= sign_coeff * s_int

                #ANNIHILATION
                self.coeffs[self.curr_it+1, :] = self.sp_coeffs
                self.coeffs[self.curr_it+1, :] += self.coeffs[self.curr_it, :]

                print(f'{self.curr_it} new spawns:      ', self.coeffs[self.curr_it+1, :], 
                    np.linalg.norm(self.sp_coeffs, ord = 1), self.S 
                )

        def run(self) -> None:
                r"""Executes the FCIQMC algorithm.
                """
                
                for i in range(self.params['it_nr']):
                        self.Nws[self.curr_it] = sum([int(np.round(np.abs(c))) for c in self.coeffs[self.curr_it, :]])
                        if i % self.params['A'] == 0 and i > self.params['delay']: 
                                self.reeval_S()					#reevaluates S to stabilize # walkers
                        self.Ss[self.curr_it+1] = self.S
                        
                        self.population_dynamics()
                        #if self.params['verbosity'] > 0:
                                #print(f'{i}. COEFFS:	', self.coeffs[self.curr_it, :])
                                #self.log.info(f'{i}. COEFFS:   {self.coeffs[self.curr_it, :]}')
                        #print(f'{i}', end='\r')
                        self.E()
                        self.curr_it += 1


def calc_mat_elem(occ_i: np.ndarray, occ_j: int, cbs: ConvolvedBasisSet, enuc: float, 
                  sao: np.ndarray, hcore: float, E_NOCI: float, overlap_ii: float = None
                  ) -> Sequence[np.ndarray]:
        r"""Outsourced calculation of Hamiltonian and 
        overlap matrix elements to parallelize code."""
        H_ij, H_ji = calc_hamiltonian(cws = occ_i, 
                                      cxs = occ_j, cbs = cbs, 
                                      enuc = enuc, holo = False,
                                      _sao = sao, _hcore = hcore)
        overlap_ij, overlap_ji = 0., 0.
        if overlap_ii is None:  
                overlap_ij, overlap_ji = calc_overlap(cws = occ_i, cxs = occ_j, cbs = cbs, 
                                                                    holo = False, _sao = sao)
                H_ij -= E_NOCI * overlap_ij
                H_ji -= E_NOCI * overlap_ji
        else:
                H_ij -= E_NOCI * overlap_ii
                H_ji -= E_NOCI * overlap_ii

        return [H_ij, H_ji, overlap_ij, overlap_ji]


