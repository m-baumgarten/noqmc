#!/usr/bin/env python
#---- author: Moritz Baumgarten ----# 
#Implementation of a nonorthogonal multireference coupled cluster
#Monte Carlo (NOCCMC) method.
#The Hilbert space for a common NOMRCI-QMC calculation is generated
#by a subset of possible exitations of the reference determinats
#(generally obtained from SCF metadynamics)
#Based on Booth, Thom and Alavi [2009], and Thom and Head-Gordon [2008]

import numpy as np
import scipy.linalg as la
from scipy.special import binom
import matplotlib.pyplot as plt
import multiprocessing
from matplotlib import rc
import sys, os, shutil
from itertools import combinations
from typing import Tuple, Sequence
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
from calc_util import (
    generate_scf, 
    number2tensor, 
    tensor2number,
    exstr2tensor,
    exstr2number,
)
from utilities import (
    Parser, 
    Log, 
    Timer,
)
from pyblock import blocking
from excips import Cluster, Excitor, flatten

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
                self.E_HF = None
                self.index_map = {}
                #if 'workdir' not in self.params:
                #        if 'output' in os.listdir():
                #                shutil.rmtree('output')
                #        os.mkdir('output')
                ##        self.params['workdir'] = os.path.join(os.getcwd(), 'output')
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
                self.E_HF = HF.e_tot
                self.enuc = HF.scf_summary['nuc']
                self.log.info(f'E_HF = {self.E_HF}')

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
                        
                        indices = [i for i in range(self.params['dim']) if self.index_map_rev[i] in [(j, ((),()), ((),())) for j in range(self.params['nr_scf'])]]
                        self.initial = np.zeros(shape=self.params['dim'], dtype=int)
                        for i in indices:
                                self.initial[i] = int(self.params['nr_w'] * self.noci_eigvecs[int(i / (self.params['dim'] / self.params['nr_scf'])), 0] / np.sum(abs(self.noci_eigvecs[:,0])))
                        
                        #NOTE GET RID:::
                        #self.initial[0] *= -1

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
                self.scf_spaces = [np.arange(self.params['dim'])[i * self.refdim : (i+1) * self.refdim] for i in range(self.params['nr_scf'])]
                self.ref_indices = [s[0] for s in self.scf_spaces]

                self.initial = np.zeros(shape = self.params['dim'], dtype = int)
                self.H = np.load('Hamiltonian.npy')
                self.H_dict = {}
                self.overlap = np.load('overlap.npy')
                self.log.info(f'Hilbert space dimensions: {self.HilbertSpaceDim}')

        def excite(self, sd: SingleDeterminant, ex: Tuple[Sequence[int],Sequence[int]] , dex: Tuple[Sequence[int],Sequence[int]]) -> SingleDeterminant:
                r"""...

                :param sd:  SingleDeterminant object used as reference determinant to create an
                            excitation space.
                :param ex:  Tuple of Sequences of MOs that  will be excited
		:param dex: Tuple of Sequences of MOs that will be deexcited

                :returns:   New SingleDeterminant object, with excited coefficient configuration."""
                new_sd = sd.copy_from(sd, dtype=np.float64)
                for i, tup in enumerate(zip(ex,dex)):
                        ex_spin, dex_spin = tup[0], tup[1]
                        if ex_spin == dex_spin: 
                                continue
                        else:
                                for to_ex, to_dex in zip(ex_spin, dex_spin): #ex_spin is either MOs in alpha or MOs in beta to be excited
                                        new_sd.coefficients[i][:, [to_ex,to_dex]] = new_sd.coefficients[i][:, [to_dex,to_ex]]
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
                for level in np.arange(0, self.params['theory_level'] + 3):   #iterates over single, double, ... excitations 
                        dim_tmp = []
                        for n_a in range(level+1):
                                n_b = int(level - n_a)
                                if n_a > n_electrons[0] or n_b > n_electrons[1]: continue
                                dim_tmp.append( int(binom(n_occ_a,n_a)) * int(binom(n_virt_a,n_a)) * int(binom(n_occ_b,n_b)) * int(binom(n_virt_b,n_b)) )               
                        self.subspace_partitioning.append(sum(dim_tmp))

                self.cumdim = np.array(self.subspace_partitioning).cumsum()            

                n_occ = n_occ_a + n_occ_b
                n_virt = n_virt_a + n_virt_b
                exs = np.sum(n_electrons)
                self.shape = np.array([len(self.reference)] + [n_occ] * exs + [n_virt] * exs + [2] * exs)

                self.log.info(f'Hilbert space dimensions for excitation levels for a SCF solution: {self.subspace_partitioning}')
                
                #define self.shape, compute all other cumulative properties as well

        def t2n(self, ex_str) -> int:
                r"""Uses tensor2number routine to generate a number corresponding to an excitation
                of the form (scf_sol, ((a_ex),(b_ex)), ((a_dex),(b_dex)))"""
                ex_lvl = len(flatten(ex_str[1]))
                n = ex_str[0] * np.sum(self.subspace_partitioning)                

                pass

        def initialize(self) -> None:
                self.initialize_references()
                self.initialize_indexmap()
                self.initialize_walkers(mode = self.params['mode'])
                self.initialize_sao_hcore()

                self.get_dimensions()

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
                self.setup()

        def setup(self):
                r"""Setup probability density for generation of Cluster sizes"""
#                #cluster_level = np.min(np.sum(self.reference[0].n_electrons), 2*self.params['theory_level'])
                self.cluster_level = np.sum(self.reference[0].n_electrons)
#                #self.p_dens = np.array([1/(2**(i+1)) for i in range(self.cluster_level-1)])
#                #rest = 1 - np.sum(self.p_dens)
#                #self.p_dens[-1] *= 2
    
        def generate_s(self) -> int:
                r"""Determines Cluster size for population dynamics of an excip."""
                return np.random.choice(a = range(1,self.cluster_level), p = self.p_dens) # TODO fix range

        def E(self) -> None:
                r"""Calculates energy estimator at current iteration."""
                
                coeffs = self.coeffs[self.curr_it,:]
                coeffs_ref = coeffs[self.ref_indices]
                overlap_tmp = np.nan_to_num(self.overlap)
                H_tmp = np.nan_to_num(self.H)
                
                index = np.where(np.abs(coeffs_ref) == np.max(np.abs(coeffs_ref)))[0][0] #get index of maximum value 
                index = self.ref_indices[index]
                E_proj = np.einsum('i,i->', H_tmp[index,:], coeffs)
                E_proj /= np.einsum('i,i->', overlap_tmp[index, :], coeffs)
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
		Writes changes to current wavefunction to sp_coeffs."""
                #NOTE Sample cluster size coeff times
                #NOTE Sample excitors corresponding to cluster sizes
                #NOTE Collapse new clusters and calculate their amplitudes, they may be outside of Hilbert space
                #NOTE Check whether they are in Hilbert space
                #NOTE Pick connected excitor (first we will do uniform prob dens)
                #NOTE Calculate corresponding matrix elements
                #NOTE   -> incorporate the fact that we may pick a double excited
                #NOTE      determinant outside Hilbert space -> not allowed to spawn on
                #NOTE      (restrict js on dets in Hilbert space)

                sp_coeffs = np.zeros(self.params['dim'], dtype = int) 
                coeffs = self.coeffs[self.curr_it, :]
                nr_excips = int(np.linalg.norm(coeffs, ord=1))
                
                #prepare parameters specific for each SCF solution
                coeffs_scf = np.array([coeffs[i*self.refdim:(i+1)*self.refdim] for i in range(self.params['nr_scf'])])
                nr_excips_scf = np.array([int(np.linalg.norm(c, ord=1)) for c in coeffs_scf]) #excludes references in probabilities
                p_scf = nr_excips_scf / nr_excips
                p_coeff_scf = np.abs(coeffs_scf) / nr_excips_scf[:, np.newaxis]   #excludes references in probabilities
                ss_scf = [min(self.cluster_level, self.refdim - 1 - len(np.isclose(c, 0).nonzero()[0])) for c in coeffs_scf]
                
                p_dens_scf = [np.array([1/(2**(i+1)) for i in range(s+1)]) for s in ss_scf]
                for p in p_dens_scf:
                        p[-1] *= 2
                
                N0_scf = np.array([c[0] for c in coeffs_scf])
                nr_excips_compl = np.array([n-N for n,N in zip(nr_excips_scf, np.abs(N0_scf))])
                #TODO no need to get p_coeff_scf, normalization of p_interm can be done with nr_excips_compl

                p_excit = 1/self.params['dim']
                #iterate over SCF sol. -> this way we pick determiants with probabilities weighted by walker pops on their SCF sol.
                for i, nr_ex in enumerate(nr_excips_scf):
                        #pick cluster sizes -> iterate over them
                        p_sel = nr_ex
                        ss = np.random.choice(a = range(ss_scf[i]+1), p = p_dens_scf[i], size = nr_ex) #could do this with expectation value -> no unique needed

                        clusters = []
                        for s in ss:
                                #generate first cluster & check whether it is 0
                                if s != 0:
                                        p_interm = p_coeff_scf[i][1:].copy()
                                        p_interm /= np.linalg.norm(p_interm, ord = 1)
                                        cluster_index = np.random.choice(np.arange(1,self.refdim), p = p_interm, replace=True, size = s) #previously False replace
                                        #p_clust2 = np.prod(np.array([np.abs(coeffs_scf[i,index]) / nr_excips_compl[i] for index in cluster_index]))
                                        #pss = np.array([np.abs(coeffs_scf[i,index]) / nr_excips_compl[i] for index in cluster_index])
                                        #p_clust2 = np.prod(pss)
                                        
                                        p_clust = np.prod(p_interm[cluster_index-1])
                                        #p_clust = 1/10
                                        #print(1/pss,  1 / p_clust, 1/p_clust2)
                                else:
                                        p_clust = 1
                                        cluster_index = np.array([0])

                                cluster_i = [self.index_map_rev[index] for index in cluster_index + i * self.refdim]
                                cluster_i = [Excitor(excitation = ex, excips = coeffs_scf[i,index]) 
                                                             for ex, index in zip(cluster_i,cluster_index)]
                                        
                                cluster_i = Cluster(excitors = cluster_i)
                                cl_ex, _ = cluster_i.collapse() #returns None, False if cluster is 0
                                ###     
                                if cl_ex is None:
                                        continue

                                cluster_i.p = p_clust
                                cluster_i.size = s
                                clusters.append(cluster_i)

                        clusters = np.array(clusters)
                        #NOTE we set ex_lvl to nelectrons instead of self.params['theory_level'] + 2
                        cluster_numbers = np.array([exstr2number(exstr = cl.excitation, shape = self.shape, ex_lvl = np.sum(self.reference[0].n_electrons)) for cl in clusters])
                        
                        clusters_len = len(clusters)
                        index_j = np.random.randint(0, self.params['dim'], size = clusters_len)
                        rand_vars = np.random.random(size = clusters_len)

                        for cluster, cl_nr, j, r in zip(clusters, cluster_numbers, index_j, rand_vars): #comes with cluster.excitation, cluster.sign, cluster.p
                                s = cluster.size
                                p_size = p_dens_scf[i][s]
                                p_clust = cluster.p * np.math.factorial(s)

                                if s == 0: amplitude = N0_scf[i]
                                else: amplitude = int(N0_scf[i]) ** (1-int(s)) * cluster.amplitude

                                the_whole_of_p = amplitude / (p_sel * p_size * p_clust * p_excit)

                                ii = None
                                if len(cluster.excitation[1][0]) + len(cluster.excitation[1][1]) <= self.params['theory_level']:
                                        ii = self.index_map[cluster.excitation]

                                        ###CACHE HAMILTONIAN###
                                if ii is not None:      #TODO switch for and ifs
                                        if np.isnan(self.H[ii,j]):
                                                det_i = self.generate_det(cluster.excitation)
                                                occ_i = det_i.occupied_coefficients
                                                det_j = self.get_det(j)
                                                occ_j = det_j.occupied_coefficients                        
                                                H_ij, _ = calc_hamiltonian(cws = occ_i,
                                                                           cxs = occ_j, cbs = self.cbs,
                                                                           enuc = self.enuc, holo = False,
                                                                           _sao = self.sao, _hcore = self.hcore)
                                                overlap_ij, _ = calc_overlap(cws = occ_i, cxs = occ_j, cbs = self.cbs,
                                                                             holo = False, _sao = self.sao)
                                                H_ij -= self.E_HF * overlap_ij
                                                self.H[ii,j] = H_ij
                                                self.overlap[ii,j] = overlap_ij
                                        else:
                                                H_ij = self.H[ii,j]
                                                overlap_ij = self.overlap[ii,j]
                                else:   #write to a Ham dirctionary, create second number of j excitation-> key will be (nr1, nr2)
                                        if (cl_nr, j) not in self.H_dict:
                                                det_i = self.generate_det(cluster.excitation)
                                                occ_i = det_i.occupied_coefficients
                                                det_j = self.get_det(j)
                                                occ_j = det_j.occupied_coefficients
                                                H_ij, _, overlap_ij, _ = calc_mat_elem(occ_i = occ_i, occ_j = occ_j, 
                                                                                       cbs = self.cbs, enuc = self.enuc, 
                                                                                       sao = self.sao, hcore = self.hcore,
                                                                                       E_HF = self.E_HF)
                                                        #H_ij -= self.E_HF * overlap_ij
                                                self.H_dict[(cl_nr, j)] = (H_ij, overlap_ij)
                                        else:        
                                                H_ij, overlap_ij = self.H_dict[(cl_nr, j)]
                                ###END OF CACHE###

                                p_spawn = self.params['dt'] * (H_ij - self.S * overlap_ij) * the_whole_of_p * cluster.sign
                                s_int = int(p_spawn)
                                b = p_spawn - s_int
                                s_int += (r < np.abs(b)) * np.sign(b)
                                sp_coeffs[j] -= s_int
                                #sp_coeffs[j] -= (r < np.abs(b)) * np.sign(b)

                                if s_int > 5:
                                        print('s_int > 5:       ', cluster.p, cluster.size, amplitude, the_whole_of_p)
                                        print('decomp:          ', amplitude, 1/p_sel, 1/p_size, 1/p_clust, 1/p_excit)
                
                #annihilation
                self.coeffs[self.curr_it+1, :] = sp_coeffs
                self.coeffs[self.curr_it+1, :] += coeffs

                print(f'{self.curr_it}. SP_COEFF:        ', self.S, np.linalg.norm(self.coeffs[self.curr_it+1, :] ,ord = 1))

        def run(self) -> None:
                r"""Executes the FCIQMC algorithm.
                """
                
                for i in range(self.params['it_nr']):
                        self.Nws[self.curr_it] = sum([int(np.round(np.abs(c))) for c in self.coeffs[self.curr_it, :]])
                        if i % self.params['A'] == 0 and i > self.params['delay']: 
                                self.reeval_S()					#reevaluates S to stabilize # walkers
                        self.Ss[self.curr_it+1] = self.S
                        
                        self.population_dynamics()
                        #print(f'{i}', end='\r')

                        self.E() #TODO Fix evaluation of projected energy
                        self.curr_it += 1
                print('Hamiltonian:     ', self.H)
                print('Overlap:         ', self.overlap)

def calc_mat_elem(occ_i: np.ndarray, occ_j: int, cbs: ConvolvedBasisSet, 
                  enuc: float, sao: np.ndarray, hcore: float, E_HF: float, 
                  overlap_ii: float = None
                  ) -> Sequence[np.ndarray]:
        r"""Outsourced calculation of Hamiltonian and 
        overlap matrix elements to parallelize code."""
        H_ij, H_ji = calc_hamiltonian(cws = occ_i, 
                                      cxs = occ_j, cbs = cbs, 
                                      enuc = enuc, holo = False,
                                      _sao = sao, _hcore = hcore)
        overlap_ij, overlap_ji = 0., 0.
        if overlap_ii is None:  
                overlap_ij, overlap_ji = calc_overlap(cws = occ_i, cxs = occ_j, 
                                                        cbs = cbs, holo = False, 
                                                        _sao = sao)
                H_ij -= E_HF * overlap_ij
                H_ji -= E_HF * overlap_ji
        else:
                H_ij -= E_HF * overlap_ii
                H_ji -= E_HF * overlap_ii

        return [H_ij, H_ji, overlap_ij, overlap_ji]

class Postprocessor(Propagator):
        r"""Class for all sorts of data 
        manipulation and evaluation tasks"""
        def __init__(self, prop: Propagator) -> None:
                self.__dict__.update(prop.__dict__)

        def get_overlap(self) -> None:
                r"""Slow and dirty method to get the full overlap matrix."""
                for i in range(self.params['dim']):
                        for j in range(self.params['dim']):
                                if i>j: continue
                                det_i = self.get_det(i)
                                det_j = self.get_det(j)
                                occ_i = det_i.occupied_coefficients
                                occ_j = det_j.occupied_coefficients
                                self.overlap[i,j], self.overlap[j,i] = calc_overlap(cws = occ_i, cxs = occ_j, cbs = self.cbs, holo = False)

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

                                processes[(i,j)] = pool.apply_async(calc_mat_elem, [occ_i, occ_j, self.cbs, self.enuc, self.sao, self.hcore, self.E_HF])

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
                        self.log.info(f'Final FCI energy:  {np.min(self.eigvals) + self.E_HF}')
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
        Li = True
        
        if Li:
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

        if not Li:
                r = 1.8
                mol = gto.M(atom=[["H", 0., 0., 0.],
                        ["H", 0., 0., r ]], basis='sto-3g', verbose=0, unit = 'Bohr') 
                mol_data = np.load('./H2_dms.npy', allow_pickle=True)[()]

        index = (np.isclose(mol_data['x'], r)).nonzero()[0][0]
        
        params = Parser().parse(sys.argv[1])
        if 'workdir' not in params:
                params['workdir'] = 'output'

        sd_rhf, sd_uhf, sd_uhf2 = generate_scf(mol, init_guess_rhf = mol_data['rhf_dm'][index], 
                                               init_guess_uhf = mol_data['uhf_dm'][index],
                                               workdir = params['workdir'])
        reference = [sd_rhf, sd_uhf, sd_uhf2]
        reference = [sd_rhf, sd_uhf]
        
        system = System(mol = mol, reference = reference, params = params)
        system.initialize()

        prop = Propagator(system)
        prop.run()

        postpr = Postprocessor(prop)
        postpr.postprocessing(benchmark = params['benchmark'])
        if params['benchmark']:
                postpr.log.info(f'good guess:     {postpr.good_guess()}')

        stat = Statistics(prop.Ss, params)
        #corr = stat.full_corr()
        #stat.blocking_stat()
        #stat.analyse()
        #error = np.array([stat.binning(n) for n in range(2,1000)])
        #np.save(os.path.join(params['workdir'], 'error.npy'), error)        
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
                #ax[0,1].set_ylim([e_corr -0.3, max(prop.E_proj[0], prop.Ss[0])])
        ax[0,1].set_ylabel('E')
        ax[0,1].set_xlabel(r'$\tau$')
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
        
#        ax[1,2].plot(stat.c0s)
#        ax[1,2].set_xlabel(r'$\tau$')
#        ax[1,2].set_ylabel(r'Conv. Param. $c_0$')

 #       ax[1,1].plot(corr)

  #      print(stat.ms)
        plt.savefig(os.path.join(params['workdir'], 'tmp.png'))
        plt.show()

