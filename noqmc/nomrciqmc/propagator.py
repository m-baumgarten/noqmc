#!/usr/bin/env python
"""---- author: Moritz Baumgarten ----# 
Implementation of a nonorthogonal multireference configuration
interaction quantum Monte Carlo (NOMRCI-QMC) method.
The Hilbert space for a common NOMRCI-QMC calculation is generated
by a subset of possible exitations of the reference determinats
(generally obtained from SCF metadynamics)
Based on Booth, Thom and Alavi [2009], and Thom and Head-Gordon [2008]
"""
import logging
import numpy as np
import os
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
from noqmc.nomrccmc.system import (
    System,
    calc_mat_elem,
)
import noqmc.nomrciqmc.excitation as excite

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
                self.Nw_ov = np.empty(self.params.it_nr, dtype = int)
                self.Nws = np.empty(self.params.it_nr, dtype = int)
                self.n = []
                self.curr_it = 0

                self.coeffs = np.zeros([self.params.it_nr+1, self.params.dim], dtype = int)
                self.coeffs[0,:] = self.initial.copy()
                print('coeff:   ', self.coeffs[0,:])

                self.E_proj = np.empty(self.params.it_nr)
                self.Ss = np.empty(self.params.it_nr)
                self.S = 0.

        def setupgenerator(self):
                r""""""
                sample = self.params.sampling
                #print(np.where(np.isnan(self.H)))
                #exit()
                if sample == 'uniform':
                        self.generator = excite.UniformGenerator(self.params)
                elif sample == 'heatbath':
                        self.generator = excite.HeatBathGenerator(self)
                elif sample == 'fragment':
                        self.generator = excite.FragmentGenerator(self)
                elif sample == 'localheatbath':
                        self.generator = excite.LocalHeatBathGenerator(self)
                elif sample == 'fragmentheatbath':
                        self.generator = excite.HeatBathFragmentGenerator(self)
                else:
                        raise NotImplementedError

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
                self.E_proj[self.curr_it] = E_proj

        def Shift(self) -> None:
                r"""Updates shift every A-th iteration.

                :param A: Interval of reevaluation of S
		:param c: Empirical daming parameter c
		"""
#                N_new = self.Nw_ov[self.curr_it]
#                N_old = self.Nw_ov[self.curr_it - self.params.A]
                N_new = self.Nws[self.curr_it]
                N_old = self.Nws[self.curr_it - self.params.A]
                self.n.append(N_new/N_old)
                self.S -= self.params.c / (self.params.A * self.params.dt) * np.log(N_new / N_old)

        def Nw(self) -> None:
                r"""Updates total number of walkers resident outside 
                the kernel of S.
                """
                overlap_tmp = np.nan_to_num(self.overlap) 
                proj = np.einsum('ij,j->i', overlap_tmp, self.coeffs[self.curr_it, :])
                self.Nw_ov[self.curr_it] = np.linalg.norm(proj, ord=1)
                self.Nws[self.curr_it] = np.linalg.norm(self.coeffs[self.curr_it, :], ord=1)

        def population_dynamics(self) -> None:
                r"""Spawning/Death in one step due to nonorthogonality. 
		Writes changes to current wavefunction to sp_coeffs.
                """
                sp_coeffs = np.zeros(self.params.dim, dtype=int)

                for i,coeff in enumerate(self.coeffs[self.curr_it, :]):
                        
                        key_i = self.index_map_rev[i]
                        det_i = self.generate_det(key_i)
                        occ_i = det_i.occupied_coefficients

                        js, pgens = self.generator.excitation(i, abs(coeff))

                        sign_coeff = np.sign(coeff)

                        set_js, index, counts = np.unique(
                            js, return_index=True, return_counts=True
                        )
                        pgens = pgens[index]

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
                                self.H[i,j], self.H[j,i] = elem[:2]
                                self.overlap[i,j], self.overlap[j,i] = elem[2:]

                        #multiplying with np.exp(-self.S) has somewhat stabilizing character
                        spawning_probs = [
                            self.params.dt * (self.H[i,j] - self.S * self.overlap[i,j]) / p
                            for j,p in zip(set_js, pgens)
                        ]
                        rand_vars = np.random.random(size=(len(spawning_probs)))
                        for j, n, ps, r in zip(set_js,counts,spawning_probs,rand_vars):
                                ps_s = n * ps
                                s_int = int(ps_s)
                                b = ps_s - s_int
                                s_int += (r < np.abs(b)) * np.sign(b)
                                sp_coeffs[j] -= sign_coeff * s_int

                                if self.params.binning:
                                        self.bin[i, j] -= sign_coeff * s_int
                                        
                        if self.params.binning:
                                for j,p in zip(set_js, pgens):
                                        self.pgen[i, j] = p
                        
                #Annihilation
                #self.coeffs[self.curr_it+1, :] = sp_coeffs
                #1+self.S
                
                coeff_tmp = np.round((1+self.S) * self.coeffs[self.curr_it, :])
                coeff_tmp = coeff_tmp + sp_coeffs
                self.coeffs[self.curr_it+1, :] = coeff_tmp

                if self.params.verbosity:
                        print(f'{self.curr_it}. Nw & S:      ',
                            np.linalg.norm(self.coeffs[self.curr_it+1, :], ord=1),
                            self.S
                        )

        def run(self) -> None:
                r"""Executes the FCIQMC algorithm.
                """
                
                self.setupgenerator()

                if self.params.binning:
                        self.bin = np.zeros_like(self.H)
                        self.pgen = np.zeros_like(self.H)

                for i in range(self.params.it_nr):
                        #Only measure Number of Walkers outside of ker(S)
                        self.Nw()
        
       #                 NORMALIZATION = i%self.params.A == 0 and i < self.params.delay/2
       #                 if NORMALIZATION:
       #                         coeff = self.coeffs[self.curr_it, :].copy()
       #                         coeff = coeff / np.linalg.norm(coeff, ord = 2)
       #                         currl1 = np.linalg.norm(coeff, ord = 1)
       #                         coeff *= self.params.nr_w
       #                         coeff /= currl1
       #                         coeff = np.round(coeff)
       #                         self.coeffs[self.curr_it, :] = coeff

#                        PROJECT = i%200 == 0 and i > 7000 and i < 7300
#                        if PROJECT:
#                                avg1 = np.mean(self.coeffs[i-450:i-400, :], axis=0)
#                                avg2 = np.mean(self.coeffs[i-50:i, :], axis=0)
#                                ideallynull = avg2-avg1
#                                print(np.einsum('ij,j->i', self.H, ideallynull))
#                                ideallynull /= np.linalg.norm(ideallynull, ord=2)
#                                projection = np.einsum('i,j->ij', ideallynull, ideallynull)
#                                print(projection)
 #                               #exit()
#
#                                coeff = self.coeffs[self.curr_it, :].copy()                                
#                                coeff = np.einsum('ij,j->i', np.eye(self.params.dim) - projection, coeff)
#                               # coeff = np.einsum('ij,j->i', projection, coeff)
#                                if np.sign(coeff[0]) != np.sign(self.coeffs[i, 0]):
#                                        coeff *= -1
#                                #coeff /= np.linalg.norm(coeff, ord = 1)
#                                #coeff *= self.Nws[i]
#                                coeff = np.round(coeff)
#                                self.coeffs[self.curr_it, :] = coeff

                        SHIFT_UPDATE = i%self.params.A == 0 and i > self.params.delay
                        if SHIFT_UPDATE: 
                                self.Shift()
                        self.Ss[i] = self.S
                        self.E()

                        self.population_dynamics()
                        
                        self.curr_it += 1


                if self.params.binning:
                        np.save(os.path.join(self.params.workdir,'bin.npy'), self.bin)
                        np.save(os.path.join(self.params.workdir,'pgen.npy'), self.pgen)



