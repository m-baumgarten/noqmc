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

                self.E_proj = np.empty(self.params.it_nr)
                self.Ss = np.empty(self.params.it_nr)
                self.S = 0

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
                N_new = self.Nw_ov[self.curr_it]
                N_old = self.Nw_ov[self.curr_it - self.params.A]
            #    N_new = self.Nws[self.curr_it]
            #    N_old = self.Nws[self.curr_it - self.params.A]
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
                self.coeffs[self.curr_it+1, :] = sp_coeffs
                self.coeffs[self.curr_it+1, :] += self.coeffs[self.curr_it, :]

                if self.params.verbosity:
                        print(f'{self.curr_it}. Nw & S:      ',
                            np.linalg.norm(self.coeffs[self.curr_it+1, :], ord=1),
                            self.S
                        )

        def heatbath_localized(self, i, size) -> np.ndarray:
                r""""""
                if size == 0:
                        return np.array([]), np.array([])
                overlap = np.zeros(self.params.dim)
                H = np.zeros_like(overlap)

                key = self.index_map_rev[i]
                det = self.generate_det(key)

                #get indices of occupied MOs
                indices = np.array([np.arange(n) for n in excited_det.n_electrons], dtype=object)
                for i, spin in enumerate(indices):
                        spin[list(key[1][i])] = list(key[2][i])
 
                #retrieve their fragment occupation
                old_ijk = [[(key[0], j, k) for k in spin] for j, spin in enumerate(indices)] 
                frags = [[self.fragmap_inv[o] for o in spin] for spin in old_ijk]
                #this will give a rule on how to generate "connected determinants" from connected MOs
                # -> (1,3) (2,1) (3,2) (4,2) 
                sample_rule = [np.unique(spin, return_counts=True) for spin in frags]

                connected_exstr = []
                for scf in range(self.params.nr_scf):
                        for s, spin in enumerate(sample_rule):
                                for (frag_i, n_i) in spin:
                                        continue
                                        #mos_on_frag = self.fragmap[]
                                        
                                        if len(self.frag_map[scf][s][frag_i]) == n_i -1:
                                                pass
                                                #excite

                                        #for all scf sols check whether there are n_i electrons that can be attributed to n_frag fragments
                                        if not len(self.frag_map[scf][s][frag_i]) >= n_i:
                                               break #break to next scf

                                        #if n_frag >= n_i generate all possible combinations of 
                                        #store all in a list of ex_str


                                connected_exstr.append()


                js = [self.index_map[ex] for ex in connected_exstr]
                #make array with self.H[i, js]
                Tij = np.abs(self.H[i,js] - self.S * self.overlap[i,js])
                pgen = Tij / np.einsum('i->', Tij)
                
                sample = np.random.choice(js, p=pgen, replace=True, size=size)
                ps = pgen[[np.where(s==j)[0][0] for s in sample]]

                return sample, ps

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



