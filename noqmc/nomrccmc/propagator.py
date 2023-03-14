#!/usr/bin/env python
"""---- author: Moritz Baumgarten ---- 
Defines the Propagator class that stochastically projects 
a CI type wave function onto its ground state.
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
from noqmc.utils.calc_util import exstr2number
from noqmc.utils.excips import (
    Cluster, 
    Excitor, 
)    

from noqmc.nomrccmc.system import System

class Propagator(System):
        r"""Class for propagation of the wavefunction/walkers in 
        imaginary time."""

        def __init__(self, system: System) -> None:
                r"""Inherits parameters from System object.

		:param system: System object containing information about Hilbert space"""
                self.__dict__.update(system.__dict__)
                self.E_proj = np.empty(self.params['it_nr'])
                self.Ss = np.empty(self.params['it_nr']+1)
                self.Nws = np.empty(self.params['it_nr'], dtype = int)
                
                self.coeffs = np.zeros(
                    [self.params['it_nr']+1, self.params['dim']], dtype = int
                )
                
                self.coeffs[0,:] = self.initial.copy()
                self.S = self.Ss[0] = 0
                self.curr_it = 0
                self.n = np.empty(self.params['it_nr'])
                self.cluster_level = np.sum(self.reference[0].n_electrons)
                self.NoneCluster = Cluster(excitors = 
                    [Excitor(
                        excitation = self.index_map_rev[0],excips = 0
                     )]
                )
                self.constS = self.params['c'] / ( self.params['A'] * self.params['dt'] )
                self.E_ref = self.E_HF

        def generate_s(self) -> int:
                r"""Determines Cluster size for population dynamics 
                of an excip."""
                return np.random.choice(a = range(1,self.cluster_level), 
                    p = self.p_dens
                ) # TODO fix range

        def E(self) -> None:
                r"""Calculates energy estimator at current iteration."""
                
                coeffs = self.coeffs[self.curr_it,:]
                coeffs_ref = coeffs[self.ref_indices]
                overlap_tmp = np.nan_to_num(self.overlap)
                H_tmp = np.nan_to_num(self.H)
                
                index = np.where(
                        np.abs(coeffs_ref) == np.max(np.abs(coeffs_ref))
                )[0][0] #get index of maximum value 
                index = self.ref_indices[index]
                E_proj = np.einsum('i,i->', H_tmp[index,:], coeffs)
                E_proj /= np.einsum('i,i->', overlap_tmp[index, :], coeffs)
                self.E_proj[self.curr_it] = E_proj

        def reeval_S(self, A: float=None, c: float=None) -> None:
                r"""Updates shift every A-th iteration.

                :param A:
		:param c:
		:param dt:"""
                N_new = self.Nws[self.curr_it]
                N_old = self.Nws[self.curr_it - self.params['A']]
                self.n[self.curr_it] = N_new / N_old
                self.S -= self.constS * np.log(N_new / N_old)


        def cluster_generation(self, nr_ex: int, ss: Sequence[int],
                p_interm: np.ndarray, scf: int, coeffs_scf: np.ndarray
                ) -> Sequence[Cluster]:
                r"""Generates a set of nr_ex Clusters corresponding to Cluster
                sizes provided in ss."""
                clusters = np.repeat(self.NoneCluster, nr_ex)              
                #count = 0
                
                for count, s in enumerate(ss):                           
                        #generate first cluster & check whether it is 0 
                        if s:                                              
                                cluster_index = np.random.choice(
                                    np.arange(1,self.refdim), p=p_interm, 
                                    replace=True, size=s
                                )
                                p_clust = np.prod(p_interm[cluster_index-1])
                        else:                                              
                                cluster_index = np.array([0])              
                                p_clust = 1                                
                                                                                   
                        cluster_i = [
                            self.index_map_rev[index] 
                            for index in cluster_index + scf * self.refdim
                        ]
                        cluster_i = [
                            Excitor(excitation=ex, excips=coeffs_scf[scf,index]) 
                            for ex, index in zip(cluster_i,cluster_index)
                        ]
                                                                                   
                        cluster_i = Cluster(excitors=cluster_i)          
                        cl_ex, _ = cluster_i.collapse() #returns None, False if cluster is 0
                                
                        if cl_ex is None:                                  
                                continue                                   
                                                                                   
                        cluster_i.p = p_clust                              
                        cluster_i.size = s                                 
                        clusters[count] = cluster_i 
                        #count += 1

                return clusters[:count]

        def cache_matelem(self, cluster: Cluster, j: int, cl_nr: int) -> Tuple:
                r"""Calculates and stores Hamiltonian and overlap matrix
                elements."""
                ii = None
                if len(cluster.excitation[1][0] + cluster.excitation[1][1]) <= self.params['theory_level']:
                        ii = self.index_map[cluster.excitation]

                #if ii is not None:
                        if np.isnan(self.H[ii,j]):
                                det_i = self.generate_det(cluster.excitation)
                                occ_i = det_i.occupied_coefficients
                                det_j = self.get_det(j)
                                occ_j = det_j.occupied_coefficients
                                
                                H_ij, _, overlap_ij, _ = calc_mat_elem(
                                    occ_i=occ_i, occ_j=occ_j, cbs=self.cbs, 
                                    enuc=self.enuc, sao=self.sao, 
                                    hcore=self.hcore, E_ref=self.E_ref
                                )
                                self.H[ii,j] = H_ij
                                self.overlap[ii,j] = overlap_ij
                        else:
                                H_ij = self.H[ii,j]
                                overlap_ij = self.overlap[ii,j]
                
                #write to a Ham dirctionary, create second number of j excitation-> key will be (nr1, nr2)
                else:
                        if (cl_nr, j) not in self.H_dict:
                                det_i = self.generate_det(cluster.excitation)
                                occ_i = det_i.occupied_coefficients
                                det_j = self.get_det(j)
                                occ_j = det_j.occupied_coefficients
                                
                                H_ij, _, overlap_ij, _ = calc_mat_elem(
                                    occ_i=occ_i, occ_j=occ_j, cbs=self.cbs, 
                                    enuc=self.enuc, sao=self.sao, 
                                    hcore=self.hcore, E_ref=self.E_ref
                                )
                                self.H_dict[(cl_nr, j)] = (H_ij, overlap_ij)
                        else:
                                H_ij, overlap_ij = self.H_dict[(cl_nr, j)]
                return H_ij, overlap_ij
                


        def population_dynamics(self) -> None:    #parallelize for each scf solution
                r"""Spawning/Death in one step due to nonorthogonality. 
		Writes changes to current wavefunction to sp_coeffs. The
                algorithm is implemented in the following way:
                        1. Sample cluster size coeff times
                        2. Sample excitors corresponding to cluster sizes
                        3. Collapse new clusters and calculate their amplitudes, 
                                they may be outside of Hilbert space
                        4. Check whether they are in Hilbert space
                        5. Pick connected excitor 
                                (first we will do uniform prob dens)
                        6. Calculate corresponding matrix elements
                                   -> incorporate the fact that we may 
                                        pick a double excited determinant 
                                        outside Hilbert space 
                                        -> not allowed to spawn on
                                          (restrict js on dets in Hilbert space)
                """

                sp_coeffs = np.zeros(self.params['dim'], dtype = int) 
                coeffs = self.coeffs[self.curr_it, :]
                nr_excips = int(np.linalg.norm(coeffs, ord=1))
                
                #prepare parameters specific for each SCF solution
                coeffs_scf = np.array(
                    [coeffs[i*self.refdim:(i+1)*self.refdim] 
                    for i in range(self.params['nr_scf'])]
                )
                nr_excips_scf = np.array(
                    [int(np.linalg.norm(c, ord=1)) for c in coeffs_scf]
                ) #excludes references in probabilities
                p_scf = nr_excips_scf / nr_excips
                p_coeff_scf = np.abs(coeffs_scf) / nr_excips_scf[:, np.newaxis]   #excludes references in probabilities
                ss_scf = [
                    min(self.cluster_level, 
                        #self.refdim - 1 - len(np.isclose(c, 0).nonzero()[0])
                        self.refdim - 1 - (c==0).nonzero()[0]
                    ) for c in coeffs_scf
                ]
                
                p_dens_scf = [
                    np.array([1/(2**(i+1)) for i in range(s+1)]) 
                    for s in ss_scf
                ]
                for p in p_dens_scf:
                        p[-1] *= 2
                
                N0_scf = np.array([c[0] for c in coeffs_scf])
                nr_excips_compl = np.array(
                    [n-N for n,N in zip(nr_excips_scf, np.abs(N0_scf))]
                )
                #TODO no need to get p_coeff_scf, normalization of p_interm can be done with nr_excips_compl

                p_excit = 1/self.params['dim']

                #iterate over SCF sol. -> this way we pick determiants 
                #with probabilities weighted by walker pops on their SCF sol.
                for i, nr_ex in enumerate(nr_excips_scf):
                        #pick cluster sizes -> iterate over them
                        p_sel = nr_ex
                        ss = np.random.choice(
                                a = range(ss_scf[i]+1), p = p_dens_scf[i], 
                                size = nr_ex
                        ) #could do this with expectation value -> no unique needed

                        p_interm = 0
                        if ss_scf[i] != 0:
                                p_interm = p_coeff_scf[i][1:].copy()
                                p_interm /= np.linalg.norm(p_interm, ord=1)

                        clusters = self.cluster_generation(
                                nr_ex=nr_ex, ss=ss, p_interm=p_interm, 
                                scf=i, coeffs_scf=coeffs_scf
                        )
                        
                        cluster_numbers = np.array([
                                exstr2number(
                                        exstr=cl.excitation, shape=self.shape, 
                                        ex_lvl=np.sum(self.reference[0].n_electrons)
                                ) 
                                for cl in clusters
                        ])
                        
                        clusters_len = len(clusters)
                        index_j = np.random.randint(0, self.params['dim'], 
                                                    size=clusters_len)
                        rand_vars = np.random.random(size = clusters_len)

                        for cluster, cl_nr, j, r in zip(clusters, cluster_numbers, index_j, rand_vars): 
                        #comes with cluster.excitation, cluster.sign, cluster.p
                                s = cluster.size
                                p_size = p_dens_scf[i][s]
                                p_clust = cluster.p * np.math.factorial(s)

                                if s == 0: amplitude = N0_scf[i]
                                else: amplitude = int(N0_scf[i]) ** (1-int(s)) * cluster.amplitude
                                #amplitude = int(N0_scf[i]) ** (1-int(s)) * cluster.amplitude if s else N0_scf[i]

                                the_whole_of_p = amplitude / (p_sel * p_size * p_clust * p_excit)

                                #calculate & store matrix elements
                                H_ij, overlap_ij = self.cache_matelem(cluster, j, cl_nr)

                                p_spawn = self.params['dt'] * (H_ij - self.S * overlap_ij) * the_whole_of_p * cluster.sign
                                
                                s_int = int(p_spawn)
                                b = p_spawn - s_int
                                s_int += (r < np.abs(b)) * np.sign(b)
                                sp_coeffs[j] -= s_int

                                if s_int > 5:
                                        print(f'Bloom detected: {s_int} on determinant {j}')
                
                #annihilation
                self.coeffs[self.curr_it+1, :] = sp_coeffs
                self.coeffs[self.curr_it+1, :] += coeffs

                print(f'{self.curr_it}. Shift and Nr_w:        ', self.S, 
                        np.linalg.norm(self.coeffs[self.curr_it+1, :] ,ord = 1)
                )

        def run(self) -> None:
                r"""Executes the population dynamics algorithm."""
                
                for i in range(self.params['it_nr']):
                        self.Nws[self.curr_it] = sum([
                            int(np.round(np.abs(c))) 
                            for c in self.coeffs[self.curr_it, :]
                        ])
                        
                        if i % self.params['A'] == 0 and i > self.params['delay']: 
                                #reevaluates S to stabilize # walkers
                                self.reeval_S()	
                        
                        self.Ss[self.curr_it+1] = self.S
                        self.population_dynamics()
                        self.E()

                        self.curr_it += 1

                print('Hamiltonian:     ', self.H)
                print('Overlap:         ', self.overlap)
                #TODO store stuff in object

def calc_mat_elem(occ_i: np.ndarray, occ_j: int, cbs: ConvolvedBasisSet, 
                  enuc: float, sao: np.ndarray, hcore: float, E_ref: float, 
                  ) -> Sequence[np.ndarray]:
        r"""Outsourced calculation of Hamiltonian and 
        overlap matrix elements to parallelize code."""
        H_ij, H_ji = calc_hamiltonian(cws = occ_i, cxs = occ_j, cbs = cbs, 
                                      enuc = enuc, holo = False, _sao = sao, 
                                      _hcore = hcore)
        
        overlap_ij, overlap_ji = calc_overlap(cws = occ_i, cxs = occ_j, 
                                              cbs = cbs, holo = False, 
                                              _sao = sao)

        H_ij -= E_ref * overlap_ij
        H_ji -= E_ref * overlap_ji

        return [H_ij, H_ji, overlap_ij, overlap_ji]


