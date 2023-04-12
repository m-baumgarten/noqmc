#!/usr/bin/env python
"""---- author: Moritz Baumgarten ---- 
General class that carries properties of the molecular system.
"""

import logging
import numpy as np
import scipy.linalg as la
from scipy.special import binom
import os
from itertools import combinations
from typing import Tuple, Sequence
from copy import deepcopy

####QCMAGIC IMPORTS
from qcmagic.core.cspace.basis.basisset import ConvolvedBasisSet
from qcmagic.core.backends.nonorthogonal_backend import (
    calc_overlap, 
    calc_hamiltonian, 
    _find_flavour_parameters,
)
from qcmagic.core.sspace.single_determinant import SingleDeterminant
from qcmagic.auxiliary.qcmagic_standards import ZERO_TOLERANCE
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
    eigh_overcomplete_noci,
    scfarray_to_state,
    E_HF,
)
from noqmc.utils.fragmentation import fragment
from noqmc.utils.utilities import Parameters

####THRESHOLDS#####
THRESHOLDS = {'ov_zero_th':       5e-06,
              'rounding':         int(-np.log10(ZERO_TOLERANCE))-4,
              }

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class System():
        r"""...
	"""
        def __init__(self, mol: Mole, params: Parameters) -> None:
                r"""A system corresponding to a provided Hamiltonian. 
		Specifies a Hilbert space basis.

		:param params: dictionary of time step, shift damping,
			       walker number, delay, verbosity and 
			       random seed"""
                assert params.delay > params.A

                self.mol = mol
                print(mol.ao_labels())
                self.params = params
                np.random.seed(self.params.seed)
                self.overlap = None     #shape dim,dim
                self.initial = None #np.empty shape dim
                self.E_NOCI = None
                self.index_map = {}
                
                logger.info(f'Arguments:      {params}')

        def get_reference(self, guess_rhf: np.ndarray, guess_uhf: np.ndarray) -> Sequence:
                r""""""
                refs = generate_scf(
                    mol=self.mol, scf_sols=self.params.scf_sols, 
                    init_guess_rhf=guess_rhf, 
                    init_guess_uhf=guess_uhf,
                    workdir=self.params.workdir,
                    localization=self.params.localization,
                )
                
                self.enuc = refs[0].scf_summary['nuc']
                self.E_HF = refs[0].e_tot
                logger.info(f'Restricted HF energy: {self.E_HF}')

                self.scfdim = len(refs[0].mo_coeff.T)
                self.refs_scfobj = refs

                self.reference = scfarray_to_state(refs)

                assert self.params.theory_level <= sum(self.reference[0].n_electrons)

                self.cbs = self.reference[0].configuration.get_subconfiguration("ConvolvedBasisSet")

        def initialize_walkers(self, mode: str = 'noci') -> None:
                r"""Generates the inital walker population on each reference
                determinant.
                
                :param mode: Specifies the type of initial guess. Default is noci.
                """

                self.initial = np.zeros(shape=self.params.dim, dtype=int)

                if mode == 'noci':
                        noci_H = np.zeros(
                            shape=(self.params.nr_scf, self.params.nr_scf)
                        )
                        noci_overlap = np.zeros_like(noci_H)
                        
                        occs = [ref.occupied_coefficients for ref in self.reference]
                        for i, occ_i in enumerate(occs):
                                for j, occ_j in enumerate(occs[i:], i):        
                                        elems = calc_mat_elem(occ_i=occ_i, occ_j=occ_j,
                                                              cbs=self.cbs, enuc=self.enuc,
                                                              sao=self.sao, hcore=self.hcore,
                                                              E_ref=0)
                                        noci_H[i,j], noci_H[j,i] = elems[:2]
                                        noci_overlap[i,j], noci_overlap[j,i] = elems[2:]
                        
                        noci_ov_eigval, noci_ov_eigvec = la.eigh(noci_overlap)
                        self.noci_eigvals, self.noci_eigvecs, _ = eigh_overcomplete_noci(noci_H, noci_overlap, noci_ov_eigval, noci_ov_eigvec)
                        self.E_NOCI = self.noci_eigvals[0]
                        
                        norm_noci = np.linalg.norm(self.noci_eigvecs[:,0], ord=1)
                        for i in range(self.params.nr_scf):        
                                nr = int(self.params.nr_w * self.noci_eigvecs[i,0] / norm_noci)
                                self.initial[self.refdim * i] = nr

                elif mode == 'ref':
                        nr = int(self.params.nr_w / self.params.nr_scf)
                        for i in range(self.params.nr_scf):
                                self.initial[self.refdim * i] = nr
                        self.E_NOCI = np.mean(E_HF(self.refs_scfobj))
                
                logger.info(f'E_NOCI = {self.E_NOCI}')
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
        
        def initialize_frag_map(self) -> None:
                r""""""
                fragments = fragment(self.sao)

                self.fragments = {i: list(frag) for i, frag in enumerate(fragments)}
                self.fragmap = [[{i: [] for i in self.fragments} for _ in range(2)] for _ in range(self.params.nr_scf)]
                self.fragmap_inv = {}

                for i, scf in enumerate(self.reference):
                        for j, coeff_spin in enumerate(scf.coefficients):
                                for k, mo in enumerate(coeff_spin.T):
                                        partial_mulliken = [np.sum(mo[frag]**2) for _, frag in self.fragments.items()]
                                        frag_index = np.where(np.max(partial_mulliken) == partial_mulliken)[0][0]
                                        
                                        self.fragmap[i][j][frag_index].append(k)
                                        self.fragmap_inv[(i,j,k)] = frag_index
                
                                #Remove empty fragments:
                                for frag in list(self.fragmap[i][j].keys()):
                                        #print(self.fragmap[i][j])
                                        if self.fragmap[i][j][frag] == []:
                                                del self.fragmap[i][j][frag]

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
                self.HilbertSpaceDim = np.zeros(self.params.theory_level + 1, dtype = int)
                for nr_scf, ref in enumerate(self.reference):
                        n_e = ref.n_electrons
                        ref_virs = ref.virtual_coefficients
                        
                        occs_alpha = range(n_e[0])
                        occs_beta = range(n_e[1])
                        virs_alpha = range(n_e[0], n_e[0]+ref_virs[0].shape[1])
                        virs_beta = range(n_e[1], n_e[1]+ref_virs[1].shape[1])
                        
                        #iterates over single, double, ... excitations 
                        for level in np.arange(0, self.params.theory_level + 1):
                                for n_alpha in range(level+1):
                                        n_beta = int(level - n_alpha)
                                        if n_alpha > n_e[0] or n_beta > n_e[1]: continue
                                        for occ_alpha in combinations(occs_alpha, n_alpha):
                                                for occ_beta in combinations(occs_beta, n_beta):
                                                        for vir_alpha in combinations(virs_alpha, n_alpha):
                                                                for vir_beta in combinations(virs_beta, n_beta):
                                                                        
                                                                        ex_tup = (nr_scf, (occ_alpha, occ_beta), (vir_alpha,vir_beta))
                                                                        self.index_map[ex_tup] = index
                                                                        index += 1
                                                                        self.HilbertSpaceDim[level] += 1

                self.index_map_rev = dict((v,k) for k,v in self.index_map.items())

                self.params.dim = int(np.sum(self.HilbertSpaceDim))
                self.refdim = self.params.dim // self.params.nr_scf

                borders = [i*self.refdim for i in range(self.params.nr_scf+1)]
                self.scf_spaces = [range(borders[i], borders[i+1]) 
                                   for i in range(self.params.nr_scf)]

                self.ref_indices = borders[:-1]
                
                if 'Hamiltonian.npy' in os.listdir():
                        self.H = np.load('Hamiltonian.npy')
                        self.overlap = np.load('overlap.npy')
                else:
                        self.H = np.full((self.params.dim, self.params.dim), np.nan)
                        print('DIM:     ', self.params.dim)
                        self.overlap = np.full((self.params.dim, self.params.dim), np.nan)
        
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
                 
                assert(n_occ_a + n_occ_b >= self.params.theory_level)
                
                self.subspace_partitioning = []
                
                #iterates over single, double, ... excitations 
                for level in np.arange(0, self.params.theory_level + 3):
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
                r""""""
                self.initialize_indexmap()
                self.initialize_sao_hcore()
                self.initialize_frag_map()
                self.get_dimensions()
                self.initialize_walkers(mode=self.params.mode)

def calc_mat_elem(occ_i: np.ndarray, occ_j: int, cbs: ConvolvedBasisSet, 
                  enuc: float, sao: np.ndarray, hcore: float, E_ref: float, 
                  ) -> Sequence[np.ndarray]:
        r"""Outsourced calculation of Hamiltonian and 
        overlap matrix elements to parallelize code."""
        H_ij, H_ji = calc_hamiltonian(cws=occ_i, cxs=occ_j, cbs=cbs, 
                                      enuc=enuc, holo=False, _sao=sao, 
                                      _hcore=hcore)
        
        overlap_ij, overlap_ji = calc_overlap(cws=occ_i, cxs=occ_j, 
                                              cbs=cbs, holo=False, 
                                              _sao=sao)

        H_ij -= E_ref * overlap_ij
        H_ji -= E_ref * overlap_ji

        return [H_ij, H_ji, overlap_ij, overlap_ji]


if __name__ == '__main__':
        #initialise a system
        pass
