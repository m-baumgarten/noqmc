#!/usr/bin/env python
"""---- author: Moritz Baumgarten ---- 
General class that carries properties of the molecular system.
"""

import logging
import numpy as np
import scipy.linalg as la
from matplotlib import rc
from scipy.special import binom
import sys, os, shutil
from itertools import combinations
from typing import Tuple, Sequence
from copy import deepcopy

####QCMAGIC IMPORTS
import qcmagic
from qcmagic.core.cspace.basis.basisset import ConvolvedBasisSet
from qcmagic.core.backends.nonorthogonal_backend import (
    calc_overlap, 
    calc_hamiltonian, 
    _find_flavour_parameters,
)
from qcmagic.core.sspace.single_determinant import SingleDeterminant
from qcmagic.auxiliary.qcmagic_standards import ZERO_TOLERANCE
from qcmagic.core.drivers.noci.basic_noci_driver import solve_noci
from qcmagic.interfaces.liqcm import (
    get_1e_ints,
    Operator,
)

###PYSCF IMPORTS
from pyscf import scf
from pyscf.gto.mole import Mole

####CUSTOM IMPORTS
from noqmc.utils.calc_util import (
    generate_scf, 
    exstr2number,
)
from noqmc.utils.utilities import (
    Parser, 
    setup_workdir,
)
from noqmc.utils.excips import (
    Cluster, 
    Excitor, 
    flatten,
)    

####THRESHOLDS#####
THRESHOLDS = {'ov_zero_th':       5e-06,
              'rounding':         int(-np.log10(ZERO_TOLERANCE))-4,
              }

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class System():
        r"""...
	"""
        def __init__(self, mol: Mole, params: dict) -> None:
                r"""A system corresponding to a provided Hamiltonian. 
		Specifies a Hilbert space basis.

		:param params: dictionary of time step, shift damping,
			       walker number, delay, verbosity and 
			       random seed"""
                assert params['delay'] > params['A']

#                if 'workdir' not in params: params['workdir'] = 'output'
#                if 'nr_scf' not in params: params['nr_scf'] = 3
#                setup_workdir(params['workdir'])

                self.mol = mol
                self.params = params
                np.random.seed(self.params['seed'])
                self.overlap = None     #shape dim,dim
                self.initial = None #np.empty shape dim
                self.E_NOCI = None
                self.index_map = {}
                
                logger.info(f'Arguments:      {params}')

        def get_reference(self, guess_rhf: np.ndarray, guess_uhf: np.ndarray) -> Sequence:
                r""""""
                refs = generate_scf(
                    mol=self.mol, scf_sols=self.params['scf_sols'], 
                    init_guess_rhf=guess_rhf, 
                    init_guess_uhf=guess_uhf,
                    workdir=self.params['workdir'],
                    localization=self.params['localization'],
                )
                
                self.reference = refs[:self.params['nr_scf']]
                
                assert self.params['theory_level'] <= sum(self.reference[0].n_electrons)

                self.cbs = self.reference[0].configuration.get_subconfiguration("ConvolvedBasisSet")
                self.params['nr_scf'] = len(self.reference)

        def initialize_references(self) -> None:
                r"""Adjusts RHF SingleDeterminant Objects to make them
                nicely excitable. Stores all reference determinants in
                a list self.reference."""
                new_refs = deepcopy(self.reference)
                for i,ref in enumerate(self.reference):
                        if len(ref.coefficients) == 1:
                                new_sd = ref.copy_from(ref, dtype=np.float64)
                                new_sd.coefficients = [new_sd.coefficients[0]] * 2
                                new_refs[i] = new_sd
                        
                        logger.info(f'Ref. {i} Coeff: {new_refs[i].coefficients}')
                self.reference = new_refs       

                HF = scf.RHF(self.mol).run()
                self.enuc = HF.scf_summary['nuc']
                self.E_HF = HF.e_tot
                
                logger.info(f'Restricted HF energy: {HF.e_tot}')

        def initialize_walkers(self, mode: str = 'noci') -> None:
                r"""Generates the inital walker population on each reference
                determinant
                
                :param mode: Specifies the type of initial guess. Default is noci.
                """
                
                if mode == 'noci':
                        noci_H = np.zeros(
                                shape=(self.params['nr_scf'], 
                                        self.params['nr_scf'])
                        )
                        noci_overlap = np.zeros(
                                shape=(self.params['nr_scf'], 
                                        self.params['nr_scf'])
                        )
                         
                        for i in range(self.params['nr_scf']):
                                det_i = self.reference[i]
                                occ_i = det_i.occupied_coefficients
                                for k in range(self.params['nr_scf'] - i):
                                        j = i+k
                                        det_j = self.reference[j]
                                        occ_j = det_j.occupied_coefficients
                                        noci_H[i,j], noci_H[j,i] = calc_hamiltonian(
                                            cws = occ_i, cxs = occ_j, 
                                            cbs = self.cbs, enuc = self.enuc, 
                                            holo = False
                                        )
                                        noci_overlap[i,j], noci_overlap[j,i] = calc_overlap(
                                            cws = occ_i, cxs = occ_j, 
                                            cbs = self.cbs, holo = False
                                        )
                        try:
                                self.noci_H = noci_H
                                self.noci_overlap = noci_overlap
                                self.noci_eigvals, self.noci_eigvecs = la.eigh(noci_H, b=noci_overlap)
                        except:
                                logger.info(
                                    f'No projection onto nonzero subspace implemented (yet).\
                                    Use distinct reference determinants or specify ref mode.'
                                )
                                raise NotImplementedError   

                        self.E_NOCI = self.noci_eigvals[0]
                        logger.info(f'E_NOCI = {self.E_NOCI}')

                        indices = self.refdim * np.arange(
                                self.params['nr_scf'], dtype=int
                        )

                        self.initial = np.zeros(shape=self.params['dim'], dtype=int)
                        
                        norm_noci = np.linalg.norm(self.noci_eigvecs[:,0], ord=1)
                        for i in indices:
                                ind1 = int(i / (self.params['dim'] / self.params['nr_scf']))
                                self.initial[i] = int(
                                    self.params['nr_w'] * self.noci_eigvecs[ind1, 0] / norm_noci
                                )

                elif mode == 'ref':
                        nr = int(self.params['nr_w'] / len(self.reference))
                        for i in range(self.params['nr_scf']):
                                self.initial[self.refdim * i] = nr
                        self.E_NOCI = self.E_HF
                logger.info(f'Initial Guess:  {self.initial}')

        def initialize_sao_hcore(self) -> None:
                r"""Calculates atomic orbital overlap matrix sao and 1e integrals
                for hamiltonian hcore."""
                cws = self.reference[0].occupied_coefficients
                nmats, nspins_per_mat, _, compl = _find_flavour_parameters(
                        [cws, cws], self.cbs
                )

                self.sao = np.kron(
                        np.identity(nspins_per_mat),
                        get_1e_ints(self.cbs, Operator.overlap, compl=compl)
                )

                self.hcore = np.kron(
                        np.identity(nspins_per_mat),
                        get_1e_ints(self.cbs, Operator.kinetic, compl=compl)
                        + get_1e_ints(self.cbs, Operator.nuclear, compl=compl)
                )

        def generate_det(self, ex_str: str) -> SingleDeterminant:
                r"""Generates Determinant at index i. Here, we only 
                implemented singly excited determinants so far.

                :param ex_str: Encodes excitation; 
                               scf_sol: integer corresponding to index in references
                               ex:      Sequence of integers corresp. to excited MOs.
                               dex:     Sequence of integers corresp. to deexcited MOs.

                :returns: New SD object corresponding to ex_str"""
                #ex is ((0,1,2,...), (0,3,5,...))
                scf_sol, ex, dex = ex_str[0], ex_str[1], ex_str[2]
                reference = self.reference[scf_sol]
                new_sd = self.excite(sd = reference, ex = ex, dex = dex)
                return new_sd

        def get_det(self, index: int) -> np.ndarray:
                r"""Returns determinant corresponding to certain index.

                :param index: index within basis set

                :returns: determinant corresponding to certain index"""
                key = self.index_map_rev[index]
                return self.generate_det(key)

        def initialize_indexmap(self) -> None:
                r"""Converts between an integer index representation of the basis 
                and an ex_str representation. It generates all determinant strings
                for a certain level of theory."""
                
                index = 0
                self.HilbertSpaceDim = np.zeros(self.params['theory_level'] + 1, dtype = int)
                for nr_scf in range(self.params['nr_scf']):
                        reference = self.reference[nr_scf]
                        n_electrons = reference.n_electrons
                        occs_alpha = range(reference.occupied_coefficients[0].shape[1])
                        occs_beta = range(reference.occupied_coefficients[1].shape[1])
                        virs_alpha = range(
                            n_electrons[0], 
                            n_electrons[0] + reference.virtual_coefficients[0].shape[1]
                        )
                        virs_beta = range(
                            n_electrons[1], 
                            n_electrons[1] + reference.virtual_coefficients[1].shape[1]
                        )
                        
                        #iterates over single, double, ... excitations 
                        for level in np.arange(0, self.params['theory_level'] + 1):
                                for n_alpha in range(level+1):
                                        n_beta = int(level - n_alpha)
                                        if n_alpha > n_electrons[0] or n_beta > n_electrons[1]: continue
                                        for occ_alpha in combinations(occs_alpha, n_alpha):
                                                for occ_beta in combinations(occs_beta, n_beta):
                                                        for vir_alpha in combinations(virs_alpha, n_alpha):
                                                                for vir_beta in combinations(virs_beta, n_beta):
                                                                        
                                                                        ex_tup = (nr_scf, (occ_alpha, occ_beta), (vir_alpha,vir_beta))
                                                                        self.index_map[ex_tup] = index
                                                                        index += 1
                                                                        self.HilbertSpaceDim[level] += 1

                self.index_map_rev = dict((v,k) for k,v in self.index_map.items())

                self.params['dim'] = int(np.sum(self.HilbertSpaceDim))
                self.refdim = self.params['dim'] // self.params['nr_scf']
                
                self.scf_spaces = [
                    np.arange(
                        self.params['dim']
                    )[i * self.refdim : (i+1) * self.refdim] 
                        for i in range(self.params['nr_scf'])
                ]
                
                self.ref_indices = [s[0] for s in self.scf_spaces]
                self.initial = np.zeros(shape = self.params['dim'], dtype = int)
                
                if 'Hamiltonian.npy' in os.listdir():
                        self.H = np.load('Hamiltonian.npy')
                        self.overlap = np.load('overlap.npy')
                else:
                        self.H = np.full((self.params['dim'], self.params['dim']), np.nan)
                        print('DIM:     ', self.params['dim'])
                        self.overlap =  np.full((self.params['dim'], self.params['dim']), np.nan)
        
                self.H_dict = {}

                logger.info(f'Hilbert space dimensions: {self.HilbertSpaceDim}')

        def excite(self, sd: SingleDeterminant, 
                   ex: Tuple[Sequence[int],Sequence[int]] , 
                   dex: Tuple[Sequence[int],Sequence[int]]
                   ) -> SingleDeterminant:
                r"""Interchanges a set of occupied MOs with a corresponding set
                of virtual MOs.

                :param sd:  SingleDeterminant object used as reference 
                            determinant to create an excitation space.
                :param ex:  Tuple of Sequences of MOs that  will be excited
		:param dex: Tuple of Sequences of MOs that will be deexcited

                :returns:   New SingleDeterminant object, with excited 
                            coefficient configuration."""
                new_sd = sd.copy_from(sd, dtype=np.float64)
                coeffs = new_sd.coefficients
                for i, tup in enumerate(zip(ex, dex)):
                        ex_spin, dex_spin = tup[0], tup[1]
                        if ex_spin == dex_spin: 
                                continue
                        #ex_spin is either MOs in alpha or MOs in beta to be excited
                        for to_ex, to_dex in zip(ex_spin, dex_spin):
                                coeffs[i][:, [to_ex,to_dex]] = coeffs[i][:, [to_dex,to_ex]]
                return new_sd

        def get_dimensions(self) -> None:
                r"""Calculates the effective Hilbert subspace dimensions, corresponding to the 
                provided excitation level + 2"""
                reference = self.reference[0]
                n_electrons = reference.n_electrons
                n_occ_a = reference.occupied_coefficients[0].shape[1]
                n_occ_b = reference.occupied_coefficients[1].shape[1]
                n_virt_a = reference.virtual_coefficients[0].shape[1]
                n_virt_b = reference.virtual_coefficients[1].shape[1]
                 
                assert(n_occ_a + n_occ_b >= self.params['theory_level'])
                
                self.subspace_partitioning = []
                
                #iterates over single, double, ... excitations 
                for level in np.arange(0, self.params['theory_level'] + 3):
                        dim_tmp = []
                        for n_a in range(level+1):
                                n_b = int(level - n_a)
                                if n_a > n_electrons[0] or n_b > n_electrons[1]: 
                                        continue
                                
                                dim_tmp.append( 
                                    int(binom(n_occ_a,n_a)) * int(binom(n_virt_a,n_a)) 
                                    * int(binom(n_occ_b,n_b)) * int(binom(n_virt_b,n_b)) 
                                )               
                        self.subspace_partitioning.append(sum(dim_tmp))

                self.cumdim = np.array(self.subspace_partitioning).cumsum()            

                n_occ = n_occ_a + n_occ_b
                n_virt = n_virt_a + n_virt_b
                exs = np.sum(n_electrons)
                self.shape = np.array(
                    [len(self.reference)] + [n_occ] * exs 
                    + [n_virt] * exs + [2] * exs
                )

                logger.info(
                    f'Hilbert space dimensions for excitation levels \
                    for a SCF solution: {self.subspace_partitioning}'
                )
 

        def initialize(self) -> None:
                self.initialize_references()
                self.initialize_indexmap()
                self.initialize_walkers(mode = self.params['mode'])
                self.initialize_sao_hcore()
                self.get_dimensions()


if __name__ == '__main__':
        #initialise a system
        pass
