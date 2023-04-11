#!/usr/bin/env python
# Author: James D Whitfield <jdwhitfield@gmail.com>
'''
Scan H2 molecule dissociation curve comparing UHF and RHF solutions per the 
example of Szabo and Ostlund section 3.8.7

The initial guess is obtained by mixing the HOMO and LUMO and is implemented
as a function that can be used in other applications.

See also 16-h2_scan.py, 30-scan_pes.py, 32-break_spin_symm.py
'''

import logging
import numpy as np
import scipy.linalg as la
from typing import Tuple, Sequence

from pyscf import fci, scf, gto, cc
from pyscf.tools.molden import from_scf
from pyscf.lo import Boys 
import os

from qcmagic.core.sspace.single_determinant import SingleDeterminant
from qcmagic.auxiliary.qcmagic_standards import ZERO_TOLERANCE
from qcmagic.interfaces.converters.pyscf import scf_to_state

from noqmc.utils.excips import flatten
from noqmc.utils.utilities import setup_workdir

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def init_guess_mixed(mol,rhf,uhf,mixing_parameter=np.pi/4):
        ''' Generate density matrix with broken spatial and spin symmetry by mixing
        HOMO and LUMO orbitals following ansatz in Szabo and Ostlund, Sec 3.8.7.
    
        psi_1a = numpy.cos(q)*psi_homo + numpy.sin(q)*psi_lumo
        psi_1b = numpy.cos(q)*psi_homo - numpy.sin(q)*psi_lumo
        
        psi_2a = -numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
        psi_2b =  numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo

        Returns: 
                Density matrices, a list of 2D ndarrays for alpha and beta spins
        '''
        # opt: q, mixing parameter 0 < q < 2 pi
    
        #based on init_guess_by_1e
        h1e = scf.hf.get_hcore(mol)
        s1e = scf.hf.get_ovlp(mol)
        mo_energy, mo_coeff = rhf.eig(h1e, s1e)
        mo_occ = rhf.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)

        homo_idx=0
        lumo_idx=1

        for i in range(len(mo_occ)-1):
                if mo_occ[i]>0 and mo_occ[i+1]<0:
                        homo_idx=i
                        lumo_idx=i+1

        psi_homo=mo_coeff[:, homo_idx]
        psi_lumo=mo_coeff[:, lumo_idx]
    
        Ca=np.zeros_like(mo_coeff)
        Cb=np.zeros_like(mo_coeff)

        #mix homo and lumo of alpha and beta coefficients
        q=mixing_parameter

        for k in range(mo_coeff.shape[0]):
                if k == homo_idx:
                        Ca[:,k] = np.cos(q)*psi_homo + np.sin(q)*psi_lumo
                        Cb[:,k] = np.cos(q)*psi_homo - np.sin(q)*psi_lumo
                        continue
                if k==lumo_idx:
                        Ca[:,k] = -np.sin(q)*psi_homo + np.cos(q)*psi_lumo
                        Cb[:,k] =  np.sin(q)*psi_homo + np.cos(q)*psi_lumo
                        continue
                Ca[:,k]=mo_coeff[:,k]
                Cb[:,k]=mo_coeff[:,k]

        dm =scf.UHF(mol).make_rdm1( (Ca,Cb), (mo_occ,mo_occ) )
        return dm 

#write generate_rhf, generate_uhf, and switch_uhf 

def generate_scf(mol, scf_sols, init_guess_rhf=None, init_guess_uhf=None, 
                 workdir='output', localization=True) -> Sequence:
        r""""""

        scf_solutions = []

        #Generate RHF solutions
        for sol in range(scf_sols[0]):
                rhf = scf.RHF(mol)
                if init_guess_rhf is None:
                        erhf = rhf.kernel()
                else:
                        erhf = rhf.kernel(init_guess_rhf)
                from_scf(rhf, os.path.join(workdir, f'rhf_{sol}.molden'))
                rhf = scf.addons.convert_to_uhf(rhf)
                rhf.mo_coeff = np.array(rhf.mo_coeff)
                scf_solutions.append(rhf)                

        #Generate UHF solutions
        for sol in range(scf_sols[1] + scf_sols[2]):
                uhf = scf.UHF(mol)
                if init_guess_uhf is None:
                        euhf = uhf.kernel(init_guess_mixed(mol, rhf, uhf))
                else:
                        euhf = uhf.kernel(init_guess_uhf)
                
                if sol >= scf_sols[1]:
                        uhf.mo_coeff[1], uhf.mo_coeff[0] = uhf.mo_coeff[0].copy(), uhf.mo_coeff[1].copy()

                from_scf(uhf, os.path.join(workdir, f'uhf_{sol}.molden'))
                scf_solutions.append(uhf)

        if localization:
                for sol in scf_solutions:
                        localize(sol)
                        #a = sol.mulliken_pop()
                        #print('Mulliken Charges:        ', a)
        
        rhf = scf.RHF(mol)
        erhf = rhf.kernel()
        dump_fci_ccsd(rhf, workdir=workdir)

        return scf_solutions

def get_MO_AO(scf_solutions: Sequence) -> dict:
        r"""scf_solutions here is an array of SingleDeterminant objects."""
        MO_AO_MAP = []
        dim = len(np.array(scf_solutions[0].coefficients).T)
        for i_sol, sol in enumerate(scf_solutions):
                MO_AO_SCF = {}
                for i_spinspace, spinspace in enumerate(sol.coefficients):
                        spinspace = np.array(spinspace.copy())

                        localized = [np.where(np.abs(mo) == np.max(np.abs(mo)))[0][0]
                                     for i_mo, mo in enumerate(spinspace.T)]
                        #signs = [int(np.sign(mo[ind])) for ind, mo in zip(localized, spinspace.T)]

                        #MO_AO_MAP.update(
                        #        {2*dim*i_sol + dim*i_spinspace + i_mo: sign*loc
                        #         for i_mo, (sign, loc) in enumerate(zip(signs, localized))}
                        #)
                        MO_AO_SCF.update(
                                {dim*i_sol + i_mo: loc
                                 for i_mo, loc in enumerate(localized)}
                        )
                MO_AO_MAP.append(MO_AO_SCF)
        #print(MO_AO_MAP)
        #exit()
        logger.info(f'MO to AO map:\n{MO_AO_MAP}')
        return MO_AO_MAP

def invert_MO_AO(MO_AO_map: dict, dim: int, nr_scf: int) -> dict:
        r""""""
        print(MO_AO_map)
     #   MO_AO_inverse = [{i: [key for key,val in MO_AO_map.items() if val==i and int(key/dim)==m] for i in range(dim*2)} for m in range(nr_scf)]
        MO_AO_inverse = None

        #for key,val in MO_AO_map.items():
        #        print(key)
        #exit()
#        MO_AO_inverse = []
#        for m in range(nr_scf):
#                tmp = {}
#                for i in range(dim):
#        #                print()
#                        tmp[i] = [key for key,val in MO_AO_map.items() if val==i] # and int(key/dim)==m]
#                MO_AO_inverse.append(tmp)

        return MO_AO_inverse


def E_HF(scf_solutions) -> Sequence[float]:
        r""""""
        return [hf.e_tot for hf in scf_solutions]
        

def scfarray_to_state(scf_solutions: Sequence) -> Sequence:
        r""""""
        return [scf_to_state(sol) for sol in scf_solutions]

def dump_fci_ccsd(mf, workdir='output') -> None:
        r""""""
        cisolver = fci.FCI(mf)
        efci = cisolver.kernel()[0]
        mycc = cc.CCSD(mf).run()
        logger.info(f'\nFCI:       {efci}\nCCSD:      {mycc.e_tot}')

def localize(mf) -> np.ndarray:
        r""""""
        occ, virt = [], []
        
        for occ_spinspace in mf.mo_occ:
                
                occ_tmp, virt_tmp = [], []
                
                for j,o in enumerate(occ_spinspace):
                        target = virt_tmp if np.isclose(o, 0.) else occ_tmp
                        target.append(j)
                
                occ.append(occ_tmp)
                virt.append(virt_tmp)
        
        #Perform localization
        for i, (o, v) in enumerate(zip(occ, virt)):        
                c_i = mf.mo_coeff[i, :, :]
                c_i[:, o] = Boys(mf.mol, c_i[:, o]).kernel() #verbose=4)
                c_i[:, v] = Boys(mf.mol, c_i[:, v]).kernel() #verbose=4)

        logger.info('localization done')

        return mf.mo_coeff

def eigh_overcomplete_noci(H, overlap, ov_eigval, ov_eigvec, loc_th=5e-06
        ) -> Tuple[np.ndarray, np.ndarray]:
        r""""""
        indices = (ov_eigval > loc_th).nonzero()[0]

                #Project onto linearly independent subspace
        projector_mat = ov_eigvec[:, indices]
        projected_ov = np.einsum(
            'ij,jk,kl->il', projector_mat.T, overlap, projector_mat
        )
        projected_ham = np.einsum(
            'ij,jk,kl->il', projector_mat.T, H, projector_mat
        )

        PRECISION = int(-np.log10(ZERO_TOLERANCE))-4
        eigvals, eigvecs = la.eigh(
            projected_ham,
            b=np.round(projected_ov, PRECISION),
            type=1
        )

        #Project back into the whole, overcomplete space whereas now the
        #eigenvectors do not have any components in the null space.
        eigvecs = np.einsum('ij,jk->ik', projector_mat, eigvecs)

        return eigvals, eigvecs, projector_mat

def tensor2number(indices: np.ndarray, shape: np.ndarray) -> int:
        r"""E.g. indices = (3,4,1) of tensor with shape (5,5,5)"""
        #number = 0
        #for i in range(len(shape)):
        #        number += indices[i] * np.prod(shape[i+1:])
        number = np.sum([ind*np.prod(shape[i+1:]) 
                         for i,ind in enumerate(indices)])
        return number


def number2tensor(number: int, shape: np.ndarray) -> np.ndarray:
        r"""Reverses tensor2number and yields the indices corresp.
        to the number provided"""
        dim = len(shape)
        indices = np.zeros(dim, dtype = np.int32)
        prod = np.prod(shape[1:])
        for i in range(dim-1):
                indices[i], number = divmod(number, prod)
                prod //= shape[i+1]
        indices[-1], number = divmod(number, prod)
        return indices


def exstr2tensor(exstr, excitation_level: int) -> np.ndarray:
        r""""""
        scf_sol = np.array([exstr[0]])
        ex_a = exstr[1][0]
        ex_b = exstr[1][1]

        spin = np.array(len(ex_a) * [0] + len(ex_b) * [1])
        flat_ex = flatten(exstr[1])
        flat_dex = flatten(exstr[2])
        fill_zeros = np.zeros( excitation_level - len(flat_ex) )
        return np.concatenate((scf_sol, flat_ex, fill_zeros, flat_dex, 
                                fill_zeros, spin, fill_zeros))

def exstr2number(exstr, shape, ex_lvl) -> int:
        r""""""
        tensor = exstr2tensor(exstr, ex_lvl)
        return tensor2number(indices = tensor, shape = shape)

if __name__ == '__main__':

        import matplotlib.pyplot as plt
    
        l = []
        euhf = []
        erhf = []
        x = np.arange(5,5.8,0.1)
        for r in x:
                mol = gto.M(atom=[["Li", 0., 0., 0.],
                    ["H", 0., 0., r]], unit='Angstrom', basis='sto-3g', verbose=0)

                _,_, e, eu, er = generate_scf(mol)
                l.append(e)
                euhf.append(eu)
                erhf.append(er)
        
        plt.plot(x, np.array(l))
        plt.plot(x, np.array(euhf))
        plt.plot(x, np.array(erhf))
        plt.savefig('b.png')
        plt.show()
