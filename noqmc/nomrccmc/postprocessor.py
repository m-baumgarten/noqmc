#!/usr/bin/env python
"""---- author: Moritz Baumgarten ---- 
Implementation of a nonorthogonal multireference coupled cluster
Monte Carlo (NOCCMC) method.
The Hilbert space for a common NOMRCI-QMC calculation is generated
by a subset of possible exitations of the reference determinats
(generally obtained from SCF metadynamics)
Based on Booth, Thom and Alavi [2009], and Thom and Head-Gordon [2008]"""

import numpy as np
import scipy.linalg as la
import multiprocessing
import sys, os, shutil
from typing import Tuple, Sequence

####QCMAGIC IMPORTS
import qcmagic
from qcmagic.core.backends.nonorthogonal_backend import (
    calc_overlap, 
    calc_hamiltonian, 
)
from qcmagic.core.sspace.single_determinant import SingleDeterminant
from qcmagic.auxiliary.qcmagic_standards import ZERO_TOLERANCE

from noqmc.nomrccmc.system import System
from noqmc.nomrccmc.propagator import Propagator
from noqmc.nomrccmc.propagator import calc_mat_elem

def mute():
    sys.stdout = open(os.devnull, 'w')


#TODO implement global thresholds, since they usually differ from revqcmagic
#thresholds

class Postprocessor(Propagator):
        r"""Class for all sorts of data 
        manipulation and evaluation tasks"""
        def __init__(self, prop: Propagator) -> None:
                #TODO predefine all stuff we want to import from prop.__dict__
                self.__dict__.update(prop.__dict__)


        def benchmark(self) -> None:
                r"""Solves the generalized eigenproblem. We project out the eigenspace 
                corresponding to eigenvalue 0 and diagonalize the Hamiltonian with
                this new positive definite overlap matrix."""
                isnan = np.isnan(self.H)
                if any(isnan.flatten()):
                        #get index of True -> evaluate H and overlap at those indices
                        
                        #TODO make sure we only have unique tuples of indices, currently we have double
                        indices = np.where(isnan)
                        
                        pool = multiprocessing.Pool(
                            processes = multiprocessing.cpu_count(), 
                            initializer = mute
                        )
                        processes = {}
                        for i,j in zip(indices[0], indices[1]):
#                                if i > j: continue
                                det_i = self.get_det(i)
                                det_j = self.get_det(j)
                                occ_i = det_i.occupied_coefficients
                                occ_j = det_j.occupied_coefficients

                                processes[(i,j)] = pool.apply_async(
                                    calc_mat_elem, 
                                    [occ_i, occ_j, self.cbs, self.enuc, self.sao, self.hcore, self.E_ref]
                                )

                        pool.close()
                        pool.join()
                        for i,j in zip(indices[0], indices[1]):
#                                if i > j: continue
                                processes[(i,j)] = processes[(i,j)].get()
                                self.H[i,j] = processes[(i,j)][0]
                                self.H[j,i] = processes[(i,j)][1]
                                self.overlap[i,j] = processes[(i,j)][2]
                                self.overlap[j,i] = processes[(i,j)][3]

 #                       for key, value in processes.items():
 #                               processes[key] = value.get()
 #                               self.H[i,j], self.H[j,i], self.overlap[i,j], self.overlap[j,i] = processes[key]

                self.ov_eigval, self.ov_eigvec = la.eigh(self.overlap)
                loc_th = 5e-06
                indices = (self.ov_eigval > loc_th).nonzero()[0]  
                
                #Project onto linearly independent subspace 
                projector_mat = self.ov_eigvec[:, indices]
                projected_ov = np.einsum(
                    'ij,jk,kl->il', projector_mat.T, self.overlap, projector_mat
                )
                
                projected_ham = np.einsum(
                    'ij,jk,kl->il', projector_mat.T, self.H, projector_mat
                )
                self.eigvals, self.eigvecs = la.eigh(
                    projected_ham, 
                    b=np.round(projected_ov,int(-np.log10(ZERO_TOLERANCE))-4), 
                    type=1
                )

                #Project back into the whole, overcomplete space whereas now the
                #eigenvectors do not have any components in the null space.
                self.eigvecs = np.einsum('ij,jk->ik', projector_mat, self.eigvecs)
                self.log.info(
                    f'Overlap Eigs:   {self.ov_eigval}, {self.ov_eigvec}\n'
                )

                self.projector1 = np.einsum('ij,jk->ik', projector_mat, projector_mat.T)

        def good_guess(self) -> np.ndarray:
                r"""Method for debugging purposes. Generates the lowest
                eigenvector, such that it may be passed on to a subsequent 
                NOCI-QMC calculation as an initial guess.

                :returns: The lowest eigen state in determinant basis."""

                ov_eigval, ov_eigvec = la.eigh(self.overlap)
                indices = (ov_eigval > 1e-10).nonzero()[0]
                ov_eigval = ov_eigval[indices]
                projector_mat = ov_eigvec[:, indices]
                ov_inv = np.diag(1 / ov_eigval)
                ov_proj_inv = np.einsum(
                    'ij,jk,lk->il', projector_mat, ov_inv, projector_mat
                )
                vec = self.eigvecs[:,0]
                return np.einsum('ij,j->i', ov_proj_inv, vec)

        def gs_degenerate(self) -> Sequence[int]:
                r"""Returns a sequence of indices, corresponing to the 
                eigenvectors that span the ground state energy eigen space 
                of our system.

                :returns: Sequence of indices specifying the degenerate ground 
                          state eigen space."""
                rounded = (
                    np.round(self.eigvals,int(-np.log10(ZERO_TOLERANCE))-12)
                )
                return (rounded == np.min(rounded)).nonzero()[0]

        def get_subspace(self, eigval: float, loc_th = 5e-06) -> np.ndarray:
                r"""Get eigenvectors corresponding to certain eigenvalues.

                :param eigval:
                :param loc_th:

                :returns:
                """
                
                if np.isclose(eigval, 0., atol = loc_th):
                        index = (np.isclose(
                            self.ov_eigval,eigval, atol = loc_th
                        )).nonzero()[0]
                        return self.ov_eigvec[:, index]
                
                if not any(np.isclose(self.eigvals,eigval)): 
                        self.log.warning(
                            f'Subspace requested for eigval {eigval}, \
                            but not present in spectrum'
                        )
                        return np.zeros_like(self.eigvecs[:,0])
                
                index = (np.isclose(self.eigvals,eigval)).nonzero()[0]
                return self.eigvecs[:, index]

        def is_in_subspace(self, subspace: np.ndarray, array: np.ndarray, 
                tolerance: float) -> bool:
                r"""Checks whether given array is in specified subspace by projecting it
                onto the (real) subspace and checking whether the (renormlized) vector is 
                left unchanged.

                :param subspace:  numpy.ndarray consisting of vectors spanning the subspace
                                  as columns.
                :param array:     array to check if in given subspace. 
                :param tolerance: Specifies tolerance when comparing array and projected array.

                :returns: Boolean indicating if array in subspace or not."""

                proj_overlap = np.einsum('ij,jk->ik', subspace.T, subspace)
                dot_coeff = np.einsum('ij,j->i', subspace.T, array)
                solve_coeff = np.linalg.solve(proj_overlap, dot_coeff)
                new = np.einsum('ij,j->i', subspace, solve_coeff) #.T 
                new /= np.linalg.norm(new, ord = 2)
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
                
                self.log.info(
                    f'final benchmark:        {self.final}, \
                    {np.linalg.norm(self.final)}'
                )
                self.log.info(
                    f'QMC final state in correct subspace? \
                    {self.is_in_subspace(subspace = subspace, array = self.coeffs[-1,:], tolerance = 1e-02)}'
                )
                
                #Projection onto different eigenstates
                if self.params['benchmark']:
                        A = np.concatenate(
                            (self.eigvecs, self.get_subspace(0.)), axis=1
                        )

                        self.proj_coeff = np.array([
                            np.linalg.solve(A, self.coeffs[i,:]) 
                            for i in range(self.coeffs.shape[0]) #TODO this is params['it_nr']
                        ])
                
        def postprocessing(self, benchmark: bool = True) -> None:
                r"""Takes care of normalisation of our walker arrays, degeneracy and dumps
                everything to the log file."""
                if benchmark:
                        self.benchmark()
                
                #Remove 0 space and normalize
                self.old_coeffs = self.coeffs.copy()
                
                self.coeffs = np.einsum('ij,kj->ik', self.projector1, self.coeffs)
                self.coeffs /= np.sqrt(np.einsum('ki,ki->i', self.coeffs, self.coeffs))
                self.coeffs = self.coeffs.T

                self.nullspace_evol = np.einsum('ij,kj->ik', np.eye(self.params['dim']) - self.projector1, self.old_coeffs)
                self.nullspace_evol /= np.sqrt(np.einsum('ki,ki->i', self.nullspace_evol, self.nullspace_evol))
                self.nullspace_evol = self.nullspace_evol.T
                
                #Selection of ground state from degenerate eigenvectors
                if benchmark:
                        self.degeneracy_treatment()
                        
                        self.log.info(
                            f'Benchmark:\nEigvals:    {self.eigvals}\n\
                            Eigvecs:        {self.eigvecs}'
                        )
                        self.log.info(
                            f'Final FCI energy:  {np.min(self.eigvals) + self.E_ref}'
                        )
                        
                        propagator = la.expm(
                            -1000 * (self.H - min(self.eigvals) * self.overlap)
                        )
                        propagated = np.einsum('ij,j->i', propagator, self.coeffs[0,:])
                        self.log.info(f'Imag. Time Evol.: {propagated/np.linalg.norm(propagated)}')

                #Dump output
                self.log.info(f'Initial Guess:  {self.initial}')
                #self.log.info(f'Final State:    {self.coeffs[-1,:]}')
                #self.log.info(f'Hamiltonian:    {self.H}')
                #self.log.info(f'Overlap:        {self.overlap}')
                #NOTE NEW


def MAE(x: np.ndarray, y: np.ndarray) -> float:
        return np.sum([np.abs(i-j) for i,j in zip(x,y)])

if __name__ == '__main__':
        pass

