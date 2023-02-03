#!/usr/bin/env python
#---- author: Moritz Baumgarten ----# 
#Implementation of a nonorthogonal multireference configuration
#interaction quantum Monte Carlo (NOMRCI-QMC) method.
#The Hilbert space for a common NOMRCI-QMC calculation is generated
#by a subset of possible exitations of the reference determinats
#(generally obtained from SCF metadynamics)
#Based on Booth, Thom and Alavi [2009], and Thom and Head-Gordon [2008]

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import multiprocessing
from matplotlib import rc
import sys, os, shutil
from itertools import combinations
from typing import (
    Tuple, 
    Sequence,
)    
from copy import deepcopy
import time

####QCMAGIC IMPORTS
import qcmagic
from qcmagic.core.cspace.basis.basisset import ConvolvedBasisSet
from qcmagic.interfaces.converters.pyscf import scf_to_state
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
    get_jk,
    Operator,
)

###PYSCF IMPORTS
from pyscf import scf, gto, fci
from pyscf.gto.mole import Mole

####CUSTOM IMPORTS
from noqmc.utils.calc_util import generate_scf
from pyblock import blocking
from noqmc.utils.utilities import (
    Parser, 
    Log, 
    Timer,
)


rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 13})
rc('text', usetex=True)


####THRESHOLDS#####
THRESHOLDS = {'ov_zero_th':       5e-06,
              'rounding':         int(-np.log10(ZERO_TOLERANCE))-4,
              }

def mute():
    sys.stdout = open(os.devnull, 'w')

class System():
        r"""...
	"""
        def __init__(self, mol: Mole, reference: list, params: dict) -> None:
                r"""A system corresponding to a provided Hamiltonian. 
		Specifies a Hilbert space basis.

		:param params: dictionary of time step, shift damping,
			       walker number, delay, verbosity and 
			       random seed"""
                assert params['delay'] > params['A']
                assert params['theory_level'] <= sum(reference[0].n_electrons)
   
                self.mol = mol
                self.reference = reference
                self.cbs = reference[0].configuration.get_subconfiguration("ConvolvedBasisSet")
                self.params = params
                self.params['nr_scf'] = len(self.reference)
                np.random.seed(self.params['seed'])
                self.overlap = None     #shape dim,dim
                self.initial = None #np.empty shape dim
                self.E_NOCI = None
                self.index_map = {}
                if 'workdir' not in self.params:
                        if 'output' in os.listdir():
                                shutil.rmtree('output')
                        os.mkdir('output')
                        self.params['workdir'] = os.path.join(
                                os.getcwd(), 'output'
                        )
                self.log = Log(filename = os.path.join(self.params['workdir'], 'log.out'))
                self.log.info(f'Arguments:      {params}')                 

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
                                #if self.params['verbosity'] > 0:
                        self.log.info(f'Ref. {i} Coeff: {new_refs[i].coefficients}')
                self.reference = new_refs       

                HF = scf.RHF(mol).run()
                self.enuc = HF.scf_summary['nuc']
                self.log.info(f'E_HF = {HF.e_tot}')

        def initialize_walkers(self, mode: str = 'noci') -> None:
                r"""Generates the inital walker population on each reference
                determinant
                
                :param mode: Specifies the type of initial guess. Default is noci."""
                
                if mode == 'noci':
                        noci_H = np.zeros(shape=(self.params['nr_scf'], self.params['nr_scf']))
                        noci_overlap = np.zeros(shape=(self.params['nr_scf'], self.params['nr_scf']))
                        for i,det_i in enumerate(self.reference):
                                for j,det_j in enumerate(self.reference):
                                        if i >=j:
                                                occ_i = det_i.occupied_coefficients
                                                occ_j = det_j.occupied_coefficients
                                                noci_H[i,j], noci_H[j,i] = calc_hamiltonian(cws = occ_i, cxs = occ_j, cbs = self.cbs, enuc = self.enuc, holo = False)
                                                noci_overlap[i,j], noci_overlap[j,i] = calc_overlap(cws = occ_i, cxs = occ_j, cbs = self.cbs, holo = False)
                        self.noci_eigvals, self.noci_eigvecs = la.eigh(noci_H, b = noci_overlap)
                      
                        self.E_NOCI = self.noci_eigvals[0]
                        self.log.info(f'E_NOCI = {self.E_NOCI}') 

                        indices = [i for i in range(self.params['dim']) if self.index_map_rev[i] in [(j, ((),()), ((),())) for j in range(self.params['nr_scf'])]]
                        self.initial = np.zeros(shape=self.params['dim'], dtype=int)
                        for i in indices:
                                self.initial[i] = int(self.params['nr_w'] * self.noci_eigvecs[int(i / (self.params['dim'] / self.params['nr_scf'])), 0] / np.sum(abs(self.noci_eigvecs[:,0])))

                elif mode == 'ref':
                        nr = int(self.params['nr_w'] / len(self.reference))
                        for i,scf_sol in enumerate(self.reference):
                                self.initial[self.index_map[(i, ((),()), ((),()))]] = nr
                
                self.log.info(f'Initial Guess:  {self.initial}')

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
                scf_sol, ex, dex = ex_str[0], ex_str[1], ex_str[2]      #where ex is ((0,1,2,...), (0,3,5,...))
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
                        virs_alpha = range(n_electrons[0], n_electrons[0] + reference.virtual_coefficients[0].shape[1])
                        virs_beta = range(n_electrons[1], n_electrons[0] + reference.virtual_coefficients[1].shape[1])
                        for level in np.arange(0, self.params['theory_level'] + 1):   #iterates over single, double, ... excitations 
                                if level == 0:
                                        ex_str = (nr_scf,((),()),((),()))
                                        self.index_map[ex_str] = index
                                        index += 1

                                        self.HilbertSpaceDim[level] += 1
                                        continue

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
                self.pdim = np.array([dim for dim in self.HilbertSpaceDim[:3]]) / np.sum(self.HilbertSpaceDim[:3])
                if self.params['theory_level'] > 1:
                        self.pdouble = self.pdim[0]/(self.pdim[0] + self.pdim[2])

                self.params['dim'] = int(np.sum(self.HilbertSpaceDim))
                self.refdim = self.params['dim'] // self.params['nr_scf'] 
                self.initial = np.zeros(shape = self.params['dim'], dtype = int)
                self.H = np.full((self.params['dim'], self.params['dim']), np.nan)
                self.overlap = np.full((self.params['dim'], self.params['dim']), np.nan)

                self.log.info(f'Hilbert space dimensions: {self.HilbertSpaceDim}')

        def excite(self, sd: SingleDeterminant, ex: Tuple[Sequence[int],Sequence[int]] , dex: Tuple[Sequence[int],Sequence[int]]) -> SingleDeterminant:
                r"""...

                :param sd:  SingleDeterminant object used as reference determinant to create an
                            excitation space.
                :param ex:  Tuple of Sequences of MOs that  will be excited
		:param dex: Tuple of Sequences of MOs that will be deexcited

                :returns:   New SingleDeterminant object, with excited coefficient configuration."""
                new_sd = sd.copy_from(sd, dtype=np.float64)
                for i, tup in enumerate(zip(ex,dex)):   #iterates over spin spaces
                        ex_spin, dex_spin = tup[0], tup[1]
                        if ex_spin == dex_spin: 
                                continue
                        else:
                                for to_ex, to_dex in zip(ex_spin, dex_spin): #ex_spin is either MOs in alpha or MOs in beta to be excited
                                        new_sd.coefficients[i][:, [to_ex,to_dex]] = new_sd.coefficients[i][:, [to_dex,to_ex]]
                return new_sd

        def initialize(self) -> None:
                self.initialize_references()
                self.initialize_indexmap()
                self.initialize_walkers(mode = self.params['mode'])
                self.initialize_sao_hcore()

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

        def reeval_S(self, A: float = None, c: float = 0.03) -> None:
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

                print(f'{self.curr_it} new spawns:      ', self.sp_coeffs, 
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
                        print(f'{i}', end='\r')
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

class Postprocessor(Propagator):
        r"""Class for all sorts of data 
        manipulation and evaluation tasks"""
        def __init__(self, prop: Propagator) -> None:
                self.__dict__.update(prop.__dict__)

        def get_overlap(self) -> None:
                r"""Quick method to get the full overlap matrix."""
                for i in range(self.params['dim']):
                        for j in range(self.params['dim']):
                                if i>j: continue
                                det_i = self.get_det(i)
                                det_j = self.get_det(j)
                                occ_i = det_i.occupied_coefficients
                                occ_j = det_j.occupied_coefficients
                                self.overlap[i,j], self.overlap[j,i] = calc_overlap(cws = occ_i, cxs = occ_j, cbs = self.cbs, holo = False)
                #self.overlap[2,5] = self.overlap[5,2] = 1.

        def benchmark(self) -> None:
                r"""Solves the generalized eigenproblem. We project out the eigenspace 
                corresponding to eigenvalue 0 and diagonalize the Hamiltonian with
                this new positive definite overlap matrix."""
                isnan = np.isnan(self.H)
                if any(isnan.flatten()):
                        #get index of True -> evaluate H and overlap at those indices
                        indices = np.where(isnan)
                        
                        pool = multiprocessing.Pool(processes = multiprocessing.cpu_count(), initializer = mute)
                        processes = {}
                        for i,j in zip(indices[0], indices[1]):
                                det_i = self.get_det(i)
                                det_j = self.get_det(j)
                                occ_i = det_i.occupied_coefficients
                                occ_j = det_j.occupied_coefficients

                                processes[(i,j)] = pool.apply_async(calc_mat_elem, [occ_i, occ_j, self.cbs, self.enuc, self.sao, self.hcore, self.E_NOCI])

                        pool.close()
                        pool.join()
                        for i,j in zip(indices[0], indices[1]):
                                processes[(i,j)] = processes[(i,j)].get()
                                self.H[i,j] = processes[(i,j)][0]
                                self.H[j,i] = processes[(i,j)][1]
                                self.overlap[i,j] = processes[(i,j)][2]
                                self.overlap[j,i] = processes[(i,j)][3]


                self.ov_eigval, self.ov_eigvec = la.eigh(self.overlap)
                loc_th = 5e-06
                indices = (self.ov_eigval > loc_th).nonzero()[0]  
                projector_mat = self.ov_eigvec[:, indices]
                projected_ov = np.einsum('ij,jk,kl->il', projector_mat.T, self.overlap, projector_mat)
                
                projected_ham = np.einsum('ij,jk,kl->il', projector_mat.T, self.H, projector_mat)
                self.eigvals, self.eigvecs = la.eigh(projected_ham, b=np.round(projected_ov,int(-np.log10(ZERO_TOLERANCE))-4), type=1) #-> assert overlap elem normalized

                self.eigvecs = np.einsum('ij,jk->ik',projector_mat,self.eigvecs)
                self.eigvecs = np.einsum('ij,jk', self.overlap, self.eigvecs)
                self.log.info(f'Overlap Eigs:   {self.ov_eigval}, {self.ov_eigvec}\n')

        def good_guess(self) -> np.ndarray:
                ov_eigval, ov_eigvec = la.eigh(self.overlap)
                indices = (ov_eigval > 1e-10).nonzero()[0]
                ov_eigval = ov_eigval[indices]
                projector_mat = ov_eigvec[:, indices]
                ov_inv = np.diag(1 / ov_eigval)
                ov_proj_inv = np.einsum('ij,jk,kl->il', projector_mat, ov_inv, projector_mat.T)
                vec = self.eigvecs[:,0]
                return np.einsum('ij,j->i', ov_proj_inv, vec)

        def gs_degenerate(self) -> Sequence[int]:
                r"""Returns a sequence of degenerate indices, corresponing to the ground
                state energy of our system."""
                rounded = (np.round(self.eigvals,int(-np.log10(ZERO_TOLERANCE))-12))
                return (rounded == np.min(rounded)).nonzero()[0]

        def get_subspace(self, eigval: float, loc_th = 5e-06) -> np.ndarray:
                r"""Get eigenvectors corresponding to certain eigenvalues"""
                if np.isclose(eigval, 0., atol = loc_th):
                        index = (np.isclose(self.ov_eigval,eigval, atol = loc_th)).nonzero()[0]
                        return self.ov_eigvec[:, index]
                if not any(np.isclose(self.eigvals,eigval)): 
                        self.log.warning(f'Subspace requested for eigval {eigval}, but not present in spectrum')
                        return np.zeros_like(self.eigvecs[:,0])
                index = (np.isclose(self.eigvals,eigval)).nonzero()[0]
                return self.eigvecs[:, index]

        def is_in_subspace(self, subspace: np.ndarray, array: np.ndarray, tolerance: float) -> bool:
                r"""Checks whether given array is in specified subspace by projecting it
                onto the (real) subspace and checking whether the (renormlized) vector is 
                left unchanged.

                :param subspace:  numpy.ndarray consisting of vectors spanning the subspace
                                  as columns.
                :param array:     array to check if in given subspace. 
                :param tolerance: Specifies tolerance when comparing array and projected array.

                :returns: Boolean indicating if array in subspace or not."""
                array /= np.linalg.norm(array)

                proj_overlap = np.einsum('ij,jk', subspace.T, subspace)
                dot_coeff = np.einsum('ij,j', subspace.T, array)
                solve_coeff = np.linalg.solve(proj_overlap, dot_coeff)
                new = np.einsum('ij,j', subspace, solve_coeff.T).T 
                new /= np.linalg.norm(new)
                return np.allclose(a = array, b = new, atol = tolerance)

        def degeneracy_treatment(self) -> None:
                r"""Determines whether degeneracy of the ground state is present. It then
                projects the final QMC state onto the eigenspace corresponding to the
                ground state."""
                degenerate_indices = self.gs_degenerate()
                subspace = self.eigvecs[:, degenerate_indices]
                
                #The following is only for the plot
                subspace_conc = np.concatenate((subspace,-subspace), axis=1)
                self.log.info(f'x&xconc: {subspace.shape, subspace_conc.shape}')
                maes = []
                for i in range(subspace_conc.shape[1]):
                        maes.append(MAE(self.coeffs[-1,:], subspace_conc[:,i]))
                self.final = subspace_conc[:, maes.index(min(maes))] 
                
                self.final /= np.linalg.norm(self.final)
                
                self.log.info(f'final benchmark:        {self.final}, {np.linalg.norm(self.final)}')
                #include 0 space:
                #if any(np.isclose(self.ov_eigval,0.)):
                #        print(subspace.shape, self.get_subspace(0.).shape, self.get_subspace(0.))
                #        subspace = np.concatenate((subspace, self.get_subspace(0.)), axis=1)
                self.log.info(f'QMC final state in correct subspace? {self.is_in_subspace(subspace = subspace, array = self.coeffs[-1,:], tolerance = 1e-02)}')
                
                #Projection onto different eigenstates
                if self.params['benchmark']:
                        #index = (np.isclose(self.ov_eigval,0)).nonzero()[0]
                        A = np.concatenate((self.eigvecs, self.get_subspace(0.)), axis=1)
                        self.proj_coeff = np.array([np.linalg.solve(A, self.coeffs[i,:]) for i in range(self.coeffs.shape[0])])
                
        def postprocessing(self, benchmark: bool = True) -> None:
                r"""Takes care of normalisation of our walker arrays, degeneracy and dumps
                everything to the log file."""
                #Normalisation
                if benchmark:
                        self.benchmark()
                self.coeffs = np.einsum('ij,jk->ik', self.overlap,self.coeffs.T).T          
                self.coeffs = np.array([self.coeffs[i,:]/np.linalg.norm(self.coeffs[i,:]) for i in range(self.params['it_nr'] + 1)])
                
                #Selection of ground state from degenerate eigenvectors
                if benchmark:
                        self.degeneracy_treatment()
                        self.log.info(f'Benchmark:\nEigvals:    {self.eigvals}\nEigvecs:        {self.eigvecs}')
                        self.log.info(f'Final FCI energy:  {np.min(self.eigvals) + self.E_NOCI}')
                        l = la.expm(-1000 * (self.H - min(self.eigvals) * self.overlap))
                        l2 = np.einsum('ij,j->i', l , self.coeffs[0,:])
               #         self.log.info(f'Imag. Time Prop.:  {l}')
                        self.log.info(f'Imag. Time Evol.: {l2/np.linalg.norm(l2)}')

                #Dump output
                self.log.info(f'Initial Guess:  {self.initial}')
                #self.log.info(f'Final State:    {self.coeffs[-1,:]}')
                #self.log.info(f'Hamiltonian:    {self.H}')
                #self.log.info(f'Overlap:        {self.overlap}')
                #NOTE NEW

class Statistics():
        r"""Class to perform statistical analysis of energy estimators (and populations later)"""
        def __init__(self, Ss: np.ndarray, params: dict) -> None:
                self.params = params
                self.Ss = Ss
                self.i = int(np.log2(self.params['it_nr'] - self.params['delay'])) - 1
                self.S = np.array( Ss[self.params['it_nr'] - 2**self.i + 1 : ] )
                self.n = len(self.S)

        def blocking_step(self, S: np.ndarray):
                r"""Performs a statistical analysis of the time averages based on the blocking method."""
                if len(S)%2!=0:
                        S = S[1:]
                S = (S[::2] + S[1::2]) / 2
                n = len(S)
                c0 = np.sqrt( (1/n) * np.var(S) )
                return np.array([c0 / (n - 1), S])

        def blocking_stat(self):
                r"""perfors the iterative blocking procedure."""
                self.c0s, self.ms = [], []
                S = self.S
                for it in range(self.i-1):
                        c0, S = self.blocking_step(S)
                        self.c0s.append(c0)
                        self.ms.append(np.mean(S))
                self.c0s = np.sqrt(np.array(self.c0s))

        def analyse(self, data: np.ndarray = None):
                if data is None: data = self.S
                data_summary = blocking.reblock(data)
                err_bound = np.max([data_summary[i][4] for i in range(len(data_summary))])
                np.save(os.path.join(self.params['workdir'], 'std_err.npy'), [data_summary[i][4] for i in range(len(data_summary))])
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
                data = np.array([self.S[m_bins*i : m_bins*(i+1)] for i in range(n_bins)]) / m_bins
                mean = np.mean(data, axis = 0)
                error = (1 / n_bins / (n_bins-1)) * np.sum( (data-mean)**2 )
                return error

        def c(self, n: int, t: int, x: np.ndarray, xbar: float) -> float:
                ct = np.einsum('i,i->', x[ : n-t] - xbar, x[t :] - xbar) / (n - t)
                return ct

        def corr(self, T: int) -> float:
                sigma2 = self.cts[0] + 2 * np.sum([(1 - t/self.n) * self.cts[t] for t in range(1,T+1)])
                sigma2 /= (self.n - 2*T - 1 + (T * (T+1) / self.n))
                return np.sqrt(np.abs(sigma2))

        def jackknifing(self):
                pass

        def full_corr(self):
                xbar = np.mean(self.S)
                self.cts = np.array([self.c(self.n, t, self.S, xbar) for t in range(self.n-1)]) #formerly self.n-1 NOTE
                corr = np.array([self.corr(t) for t in range(1, self.n-1)]) #formerly self.n-1 NOTE
                
                plt.plot(self.cts)
                plt.savefig(os.path.join(self.params['workdir'],'cts.png'))
                plt.close()
                i = int(np.log2(self.params['it_nr'] - self.params['delay']))
                S = np.array( self.Ss[self.params['it_nr'] - 2**i + 1 : ] )
                xbar = np.mean(S)
                cts = np.array([self.c(len(S), t, S, xbar) for t in range(1, len(S)-1)])
                plt.plot(cts)
                plt.savefig(os.path.join(self.params['workdir'],'long_cts.png'))
                plt.close()
                return corr

class Diagonalizer(Propagator): 
        r'''Diagonalizer should take Propagator objects for each eigenvector we're aiming 
	to find and evolve them simultaneously, while orthogonalizing coeffs (each time 
	step?)
	Use multiprocessing here. Join processes after each time step.''' 
        def __init__(self) -> None:
                #matrix of coeffs, columns are coeff for one state
                pass

	

        def run(self) -> None:
                #for state in 	
                pass	



def MAE(x: np.ndarray, y: np.ndarray) -> float:
        return np.sum([np.abs(i-j) for i,j in zip(x,y)])

if __name__ == '__main__':
        
        #MOLECULAR SYSTEM
        r = 3.
        basis = {'H': gto.basis.parse('''
        H       S
                13.01000        0.01968500
                1.962000        0.1379770
                0.444600        0.4781480
                0.122000        0.5012400'''),
                'Li': gto.basis.parse('''
        Li      S
                1469.000        0.0007660       -0.0001200
                220.5000        0.0058920       -0.0009239
                50.26000        0.0296710       -0.0046890
                14.24000        0.1091800       -0.0176820
                4.581000        0.2827890       -0.0489020
                1.580000        0.4531230       -0.0960090
                0.564000        0.2747740       -0.1363800
                0.073450        0.0097510       0.57510200
                0.028050        -0.003180       0.51766100
        Li      P
                1.534000        0.0227840       0.00000000
                0.274900        0.1391070       0.00000000
                0.073620        0.5003750       0.00000000
                0.024030        0.5084740       1.00000000''')}
        
        mol = gto.M(atom=[["Li", 0., 0., 0.],
                        ["H", 0., 0., r ]], basis = basis, verbose = 0, unit = 'Angstrom') 
        mol_data = np.load('./LiH_dms.npy', allow_pickle=True)[()]
        
        index = (np.isclose(mol_data['x'], r)).nonzero()[0][0]
        sd_rhf, sd_uhf, sd_uhf2 = generate_scf(mol, init_guess_rhf = mol_data['rhf_dm'][index], 
                                               init_guess_uhf = mol_data['uhf_dm'][index])
        reference = [sd_rhf, sd_uhf, sd_uhf2]
        reference = [sd_rhf, sd_rhf]
        params = Parser().parse(sys.argv[1])
        
        system = System(mol = mol, reference = reference, params = params)
        system.initialize()

        prop = Propagator(system)
        prop.run()

        postpr = Postprocessor(prop)
        postpr.postprocessing(benchmark = params['benchmark'])
        if params['benchmark']:
                postpr.log.info(f'good guess:     {postpr.good_guess()}')

        stat = Statistics(prop.Ss, params)
        corr = stat.full_corr()
        stat.blocking_stat()
        stat.analyse()
        error = np.array([stat.binning(n) for n in range(2,1000)])
        np.save(os.path.join(params['workdir'], 'error.npy'), error)        
        np.save(os.path.join(params['workdir'], 'Ss.npy'), prop.Ss)
        np.save(os.path.join(params['workdir'], 'Hamiltonian.npy'), prop.H)
        np.save(os.path.join(params['workdir'], 'overlap.npy'), prop.overlap)

        #PLOT
        fig,ax = plt.subplots(2,3, figsize=(14,6))
        plt.subplots_adjust(wspace = 0.4)

        #PLOT 1: Coefficients obtained from QMC
        for i in range(system.params['dim']):
                ax[0,0].plot(np.arange(params['it_nr'] + 1) * params['dt'], 
                           postpr.coeffs[:,i],color = f'C{i}', label=f'{i}')
                if params['benchmark']:
                        ax[0,0].hlines(postpr.final[i],
                                (postpr.params['it_nr']+1) * params['dt'], 0, color=f'C{i}', linestyle = 'dashed')
                ax[0,0].set_ylabel('Ground State Coeff.')
                ax[0,0].set_xlabel(r'$\tau$')
        ax[0,0].legend(frameon=False)

        #PLOT 2: Energy estimator, Shift and E_C from FCI
        ax[0,1].plot(np.arange(params['it_nr']) * params['dt'], prop.E_proj,label=r'$E(\tau)$')
        ax[0,1].plot(np.arange(params['it_nr'] + 1) * params['dt'], prop.Ss,label=r'$S(\tau)$')
        if params['benchmark']:
                ax[0,1].hlines(np.min(postpr.eigvals) , prop.params['it_nr'] * params['dt'], 
                             0, color='black', linestyle = 'dashed',label=r'$E_{FCI}$')
                e_corr = np.min(postpr.eigvals)
#                ax[1].set_ylim([e_corr -0.3, max(prop.E_proj[0], prop.Ss[0])])
        ax[0,1].set_ylabel('E')
        ax[0,1].set_xlabel(r'$\tau$')
        ax[0,1].legend(frameon=False)
        ax[0,1].legend(frameon=False)

        if params['benchmark']:
                for i in range(system.params['dim']):
                        ax[1,0].plot(np.arange(params['it_nr'] + 1) * params['dt'], 
                        postpr.proj_coeff[:,i], color = f'C{i}', label=f'{i}')
                        
                        ax[1,0].set_ylabel('Contrib. to Coeff')
                        ax[1,0].set_xlabel(r'$\tau$')
                ax[1,0].legend(frameon=False)

        ax[0,2].plot(np.arange(params['it_nr']) * params['dt'], prop.Nws)
        ax[0,2].set_xlabel(r'$\tau$')
        ax[0,2].set_ylabel('Total Number of Walkers')
        
        ax[1,2].plot(stat.c0s)
        ax[1,2].set_xlabel(r'$\tau$')
        ax[1,2].set_ylabel(r'Conv. Param. $c_0$')

        ax[1,1].plot(corr)

        print(stat.ms)
        plt.savefig(os.path.join(params['workdir'], 'tmp.png'))
        plt.show()

