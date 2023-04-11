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
from scipy.special import binom
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
                self.E_ref = self.E_HF

                self.coeffs = np.zeros([self.params.it_nr+1, self.params.dim], dtype = int)
                self.coeffs[0,:] = self.initial.copy()

                self.E_proj = np.empty(self.params.it_nr+1)
                self.E_proj[0] = self.E_NOCI - self.E_ref

                self.Ss = np.empty(self.params.it_nr+1)
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
                N_new = self.Nw_ov[self.curr_it]
                N_old = self.Nw_ov[self.curr_it - self.params.A]
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
                       
                        #excited_str, pgens = self.excitation(ex_str=self.index_map_rev[i], size=abs(coeff))

                        sign_coeff = np.sign(coeff)
                        
                        #Uniform Excitation Generation -> TODO Generalize
                        js = np.random.randint(0, self.params.dim, size=abs(coeff))

                        #Prepare self.H and self.overlap:
                        key_i = self.index_map_rev[i]
                        det_i = self.generate_det(key_i)
                        occ_i = det_i.occupied_coefficients
                        
                        set_js, index, counts = np.unique(
                            js, return_index=True, return_counts=True
                        )

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
                            self.params.dim * self.params.dt 
                            * (self.H[i,j] - self.S * self.overlap[i,j])
                            for j in set_js
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
                                        self.pgen[i,j] = 1/self.params.dim
                #Annihilation
                self.coeffs[self.curr_it+1, :] = sp_coeffs
                self.coeffs[self.curr_it+1, :] += self.coeffs[self.curr_it, :]

                if self.params.verbosity:
                        print(f'{self.curr_it}. Nw & S:      ',
                            np.linalg.norm(self.coeffs[self.curr_it+1, :], ord=1), 
                            self.S 
                        )

        def pop_dynamics_exc(self) -> None:
                r""""""
                sp_coeffs = np.zeros(self.params.dim, dtype=int)

                for i,coeff in enumerate(self.coeffs[self.curr_it, :]):
                        
                        excited_str, pgens = self.excitation(ex_str=self.index_map_rev[i], size=abs(coeff))
                        js = [self.index_map[s] for s in excited_str]

                        sign_coeff = np.sign(coeff)

                        key_i = self.index_map_rev[i]
                        det_i = self.generate_det(key_i)
                        occ_i = det_i.occupied_coefficients

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
                                        if self.pgen[i, j] != p and self.pgen[i, j] != 0:
                                                print('pgen & p:', self.pgen[i, j], p)
                                        self.pgen[i, j] = p
                        

                        ##REMOVE THIS TODO
                      #  for i in range(self.params['dim']):
                      #  
                      #          excited_str, pgens = self.excitation(ex_str=self.index_map_rev[i], size=100)
                      #          js = [self.index_map[s] for s in excited_str]
                      #          set_js, index, counts = np.unique(
                      #                  js, return_index=True, return_counts=True
                      #          )
                      #          pgens = pgens[index]
                      #          for j,p in zip(set_js, pgens):
                      #                  self.pgen[i, j] = p
                      #          np.save(os.path.join(self.params['workdir'],'pgen.npy'), self.pgen)
                      #  exit()
                        #########

                #Annihilation
                self.coeffs[self.curr_it+1, :] = sp_coeffs
                self.coeffs[self.curr_it+1, :] += self.coeffs[self.curr_it, :]

                if self.params.verbosity:
                        print(f'{self.curr_it}. Nw & S:      ',
                            np.linalg.norm(self.coeffs[self.curr_it+1, :], ord=1),
                            self.S
                        )
                pass

        def excitation(self, ex_str: Tuple=(1, ((0,),()), ((4,),())), size: int=1): #-> Sequence:
                r"""
                :params det: Determinant we want to excite from
                :params nr:  Number of excitations we wish to generate
                
                :returns: array of keys and array of generation probabilities"""
                #Potentially exchange this with SCF transition matrix 
                if size==0:
                        return [], np.array([])

                scf_spawn = np.random.randint(0, self.params.nr_scf, size=size)
                # 1. Prepare MO list for ex_str:
                excited_det = self.generate_det(ex_str)
                # indices will contain the MO indices corresponding to an ex_str, 
                # conserving the spin structure of the ex_str
                indices = np.array([np.arange(n) for n in excited_det.n_electrons], dtype=object)
                for i, spin in enumerate(indices):
                        spin[list(ex_str[1][i])] = list(ex_str[2][i])
                
                out = []
                p = []
                for s in scf_spawn:
                        pgen = 1./self.params.nr_scf
                        
                        if s == ex_str[0]:
                                #Let's first try with a uniform sampling scheme:
                                pgen *= 1./self.refdim
                                index = s * self.refdim + np.random.randint(self.refdim)
                                new_str = self.index_map_rev[index]
              
                        else:
                                old_ijk = [[(ex_str[0], j, k) for k in spin] for j, spin in enumerate(indices)] 
                                frags = [[self.fragmap_inv[o] for o in spin] for spin in old_ijk]
                                sample_rule = [np.unique(spin, return_counts=True) for spin in frags]

                                # 2. sample MOs according to sample_rule and produce correct pgen
                                mo_sspace, pgen_mo, localized_on_frag = self.sample_mos(scf=s, rule=sample_rule)
                               
                                if not localized_on_frag:
                                        continue

                                pgen *= pgen_mo
                                dexcits, excits, allowed = self.collapse(mo_sspace, excited_det.n_electrons)
                                #print(dexcits, excits, allowed)

                                if not allowed: 
                                        continue
                                
                                new_str = (s, dexcits, excits)
                        
                        out.append(new_str)
                        p.append(pgen)
                
                return out, np.array(p)

        def sample_mos(self, scf: int, rule: np.ndarray) -> (list, float, bool):
                r""""""
                mo_sspace = []
                localized_on_frag = True
                pgen = 1.

                for i_s, spin in enumerate(rule):
                        
                        mo = []
                        ind, freq = spin
                        frags = self.fragmap[scf][i_s]                                       
                                
                        (ind, freq), ploc = self.excite_local(ind, freq)
                        pgen *= ploc
                        #print(ind, freq)
                        #Ensure necessary amount of electrons is localized on all 
                        #necessary fragments goverened by the determinant we spawn from
                        #print(frags, ind, freq)
                        for j,i in enumerate(ind):
                                if i not in frags:
                                        return [], 0.0, False
                                if len(frags[i]) < freq[j]:
                                        return [], 0.0, False

                        for i, n in zip(ind, freq):
                                mo.append(np.random.choice(frags[i], size=n, replace=False))
                                pgen *= 1/binom(len(frags[i]), n)
                        
                        mo = np.concatenate(mo)
                        mo_sspace.append(mo)
 
                return mo_sspace, pgen, True

        def excite_local(self, ind: np.ndarray, freq:np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
                r"""Decides whether to excited one electron to a nearest neighbor fragment
                and returns the resulting new fragment configuration with corresponding
                frequencies."""
                dont_excite = np.round(np.random.random())   #create binary number
                
                if dont_excite:
                        return (ind, freq), 0.5
                
                localized = np.repeat(ind, freq)
                l = len(localized)
                ploc = 1/l
                excited = np.random.choice(range(l))  #is sum(freq) faster?                              
                localized[excited] += 1 - 2 * int(np.round(np.random.random()))
                
                return np.unique(localized, return_counts=True), 0.5*ploc

        def collapse(self, indices: list, n_e: list) -> (list, bool):
                r"""Evaluates whether a set of MO indices is within the Hilbert space
                defined by the maximum allows excitation level."""
                dex = [[],[]]
                ex = [[],[]]
                for j, (ind, n) in enumerate(zip(indices, n_e)):
                        dex[j] = tuple(set(range(n)).difference(ind))
                        for m, i in enumerate(ind):
                                if i >= n:
                                        ex[j].append(i)
                for spin in ex:
                        spin.sort()
                ex = [tuple(spin) for spin in ex]
                level = sum([len(s) for s in ex]) 
                return tuple(dex), tuple(ex), level <= self.params.theory_level

        def run(self) -> None:
                r"""Executes the FCIQMC algorithm.
                """
                
                pop_dyn = self.population_dynamics
                if not self.params.uniform:
                        assert(self.params.localization)
                        pop_dyn = self.pop_dynamics_exc
                        logger.info('Using localized excitation generation scheme!')

                if self.params.binning:
                        self.bin = np.zeros_like(self.H)
                        self.pgen = np.zeros_like(self.H)

                for i in range(self.params.it_nr):

                        #Only measure Number of Walkers outside of ker(S)
                        self.Nw()
                        
                        SHIFT_UPDATE = i%self.params.A == 0 and i > self.params.delay
                        if SHIFT_UPDATE: 
                                self.Shift()
                        self.Ss[self.curr_it+1] = self.S

                        pop_dyn()

                        self.E()
                        self.curr_it += 1

                if self.params.binning:
                        np.save(os.path.join(self.params.workdir,'bin.npy'), self.bin)
                        np.save(os.path.join(self.params.workdir,'pgen.npy'), self.pgen)
