from itertools import combinations
import logging
import numpy as np
from scipy.special import binom
from noqmc.nomrccmc.system import System
from noqmc.utils.utilities import Parameters
from typing import Sequence, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ExcitationGenerator(System):
        r""""""
        def __init__(self, parameters: Parameters):
                self.params = parameters

        def excitation(self, i: int, size: int) -> (np.ndarray, np.ndarray):
                r"""Each derivative of ExcitationGenerator must have an
                excitation method, taking determinant number & size of 
                generation into account. """
                return np.array([]), np.array([])

class UniformGenerator(ExcitationGenerator):
        r""""""
        def __init__(self, parameters: Parameters):
                self.dim = parameters.dim
                self.p = 1/parameters.dim

        def excitation(self, i: int, size: int) -> (np.ndarray, np.ndarray):
                js = np.random.randint(0, self.dim, size=size)
                pgens = np.repeat(self.p, size)
                return js, pgens


class OrthogonalGenerator(ExcitationGenerator):
        r""""""
        def __init__(self):
                pass

        def excitation(self, i: int, size: int) -> (np.ndarray, np.ndarray):
                ex_str = self.index_map[i]
                
                #determine nr of singles/doubles with expectation value
                pass

        def double(self, size: int):
                pass

        def single(self, size: int):
                pgen = 1.
                # 1. pick spin space
                
                p = 1/2
                # 2. uniformly pick i from occ orbs
                # 3. uniformly pick j from virt orbs
                pass

class OrthogonalHeatBathGenerator(OrthogonalGenerator):
        r""""""
        def __init__(self):
                pass

class HeatBathGenerator(ExcitationGenerator):
        r""""""
        def __init__(self, propagator: dict) -> None:
                self.validate(propagator)
                self.prop = propagator

        def excitation(self, i: int, size: int) -> (np.ndarray, np.ndarray):
                r""""""
                if size == 0:
                        return np.array([]), np.array([])

                Ti = np.abs(self.prop.H[i,:] - self.prop.S * self.prop.overlap[i,:])  
                p_distrib = Ti / np.einsum('i->', Ti) 
                js = np.random.choice(np.arange(self.prop.params.dim, dtype=int), 
                                      p=p_distrib, replace=True, size=size)
                pgen = p_distrib[js] 
                
                return js, pgen
               
        def validate(self, propagator: dict):
                
                H = propagator.H
                overlap = propagator.overlap
                HAMILTONIAN_PRECOMPUTED = not any(np.isnan(H).flatten())
                OVERLAP_PRECOMPUTED = not any(np.isnan(overlap).flatten())

                if not HAMILTONIAN_PRECOMPUTED or not OVERLAP_PRECOMPUTED:
                        propagator._calculatematrices()

class LocalHeatBathGenerator(HeatBathGenerator, OrthogonalGenerator):
        r""""""
        def __init__(self, propagator: dict):
                super().__init__(propagator)
                self.__dict__.update(propagator.__dict__)
                self.initialize_allowed()
                self.update_allowed()

        def det2frags(self, i: int):
                r"""Maps a determinant index, as in the index_map to a set of 
                fragments."""
                ex_str = self.index_map_rev[i]
                excited_det = self.generate_det(ex_str)
                indices = np.array([np.arange(n) for n in excited_det.n_electrons], dtype=object)
                for s, spin in enumerate(indices):
                        spin[list(ex_str[1][s])] = list(ex_str[2][s])
                #print(ex_str, indices)

                old_ijk = [[(ex_str[0], j, k) for k in spin] for j, spin in enumerate(indices)]
                frags = [[self.fragmap_inv[o] for o in spin] for spin in old_ijk]
                frag_occupation = [np.unique(spin, return_counts=True) for spin in frags]
               
                local_representation = []
                for i_s, spin in enumerate(frag_occupation):
                        ind, freq = spin
                        localized = tuple(np.sort(np.repeat(ind, freq)))
                        local_representation.append(localized)       
                
                return tuple(local_representation)


        def mapdets(self):
                self.detmap = {i: self.det2frags(i) for i in range(self.prop.params.dim)}
                
                self.detmap_inv = {}
                for i,fspin in self.detmap.items():
                        if fspin in self.detmap_inv:
                                self.detmap_inv[fspin].append(i)
                        else:
                                self.detmap_inv[fspin] = [i]

        def initialize_allowed(self):
                self.mapdets()
                
                self.allowed = np.zeros((self.prop.params.dim, self.prop.params.dim), dtype=int)
                for i, frag in self.detmap.items():
                        allowed = self.detmap_inv[frag]
                        for det in allowed:
                                self.allowed[i, det] = 1
                #this currently only sets allowed to one where frags have same occupation
                assert((self.allowed == self.allowed.T).all())
                
                #self.allowed = np.kron(np.ones((2,2), dtype=int), self.allowed[:118,:118])
                
                np.save('allowed_init.npy', self.allowed)

        def update_allowed(self):
                r"""Updates the self.allowed look-up array."""
                for i in self.detmap:
                        
                        js = np.where(self.allowed[i,:])[0]
                        #ex_strs = [self.index_map_rev[j] for j in js]
               
                        #the following will look a lot like initialize_indexmap,
                        #but using the js as references. we will then compare
                        #resulting ex_str's and check that they lie in our
                        #Hilbert space
                        for j in js:
                                ex_str = self.index_map_rev[j]
                                excited_det = self.generate_det(ex_str)
                                n_e = excited_det.n_electrons
                                indices = self.getindices(ex_str)
                                
                                for ex in self.index_map:
                                        if ex[0] != ex_str[0]: continue
                                        excited_indices = self.swap_mos(indices, ex)
                                        new_ex_str = self.get_ex_str(excited_indices, n_e, ex[0])
                                        if new_ex_str in self.index_map:
                                                k = self.index_map[new_ex_str]
                                                self.allowed[i,k] = 1
                                                

                self.allowed += self.allowed.T
                np.save('allowed.npy', self.allowed)

        def swap_mos(self, indices, ex_str):
                r"""Indices must be the full order of MOs, including occupied and virtual ones.
                Swaps set of indices accoring to permutation characteristic to ex_str"""
                new_indices = indices.copy()
                scf_sol, ex, dex = ex_str[0], ex_str[1], ex_str[2]
                for s, spin in enumerate(zip(ex, dex)):
                        ex_spin, dex_spin = spin[0], spin[1]
                        if ex_spin == dex_spin:
                                continue
                        for to_ex, to_dex in zip(ex_spin, dex_spin):
                               new_indices[s][[to_ex,to_dex]] = new_indices[s][[to_dex,to_ex]]
                return new_indices


        def get_ex_str(self, indices, n_es, scf):
                r"""indices is an permuted set of indices and we want
                to find the corresponding ex_str"""

                occs_tot = []
                virt_tot = []
                for s, spin in enumerate(indices):
                        n_e = n_es[s]
                        occ = set(spin[:n_e])
                        ref = set(range(n_e))
                        diff = np.array(list(occ.symmetric_difference(ref)))
                        
                        #sort it and slice in bits of occ and virt indices
                        diff = np.sort(diff)
                        occs_tot.append(tuple(diff[np.where(diff < n_e)[0]]))
                        virt_tot.append(tuple(diff[np.where(diff >= n_e)[0]]))
                        
                new_ex_str = (scf, (occs_tot[0], occs_tot[1]), (virt_tot[0], virt_tot[1]) )
                return new_ex_str

        def getindices(self, ex_str):
                excited_det = self.generate_det(ex_str)
                indices = np.array([np.arange(len(coeff)) for coeff in excited_det.coefficients], dtype=object)
                indices = self.exciteindices(indices, ex_str)
                return indices

        def exciteindices(self, indices, ex_str):
                for i, spin in enumerate(indices):
                        ex = list(ex_str[1][i])
                        dex = list(ex_str[2][i])
                        ex, dex = dex, ex
                        spin[ex], spin[dex] = spin[dex], spin[ex]
                return indices

        def excitation(self, i: int, size: int) -> (np.ndarray, np.ndarray):
                r""""""
                if size==0: #write decorator for this?
                        return [], np.array([])

                # 2. calculate Tij's for all j's and their single & double excitations.
                #    This is here encoded in self.allowed, which we precompute upon initialization
                #    of this class.
                Ti = np.abs(self.prop.H[i,:] - self.prop.S * self.prop.overlap[i,:])
                Ti = np.where(self.allowed[i,:], Ti, 0.)
                
                p_distrib = Ti / np.einsum('i->', Ti) 
                js = np.random.choice(np.arange(self.prop.params.dim, dtype=int), 
                                      p=p_distrib, replace=True, size=size)
                pgen = p_distrib[js] 
                
                return js, pgen


class HeatBathFragmentGenerator(LocalHeatBathGenerator):
        r"""This is an approximation to the general heat-bath generation scheme.
        Tij's that are connected via fragments and their single excitations
        are approximated to be 0. 
        Fun fact: It doesn't work, at least for overlap based fragmentation 
                  when looking at LiH."""
        def __init__(self, propagator: dict):
                self.validate(propagator)
                self.prop = propagator
                self.__dict__.update(propagator.__dict__)
                self.initialize_allowed()
                self.update_allowed()


        def update_allowed(self):
                r"""Updates the self.allowed look-up array."""
                for i in self.detmap:
                        
                        js = np.where(self.allowed[i,:])[0]
                        frag_config = self.detmap[js[0]]
                                
                        excited = [[],[]]
                        for s, frag_spin in enumerate(frag_config):
                                
                                for index,f in enumerate(frag_spin):
                                        for excitation in [1,-1]:
                                                if f==0 and excitation==-1:
                                                        continue
                                                if f==len(self.fragments)-1 and excitation==1:
                                                        continue
                                                frag_tmp = list(frag_spin).copy()
                                                frag_tmp[index] += excitation
                                                excited[s].append(tuple(frag_tmp))

                        excited.reverse()
                        new_js = []
                        for i, (prev, spin) in enumerate(zip(frag_config, excited)):
                                for frag in spin:
                                        if i:
                                                new = (frag, prev)
                                        else:
                                                new = (prev, frag)
                                        if new in self.detmap_inv:
                                                new_js.append(self.detmap_inv[new])

                        for j in new_js:
                                self.allowed[i,j] = 1

class FragmentGenerator(ExcitationGenerator):
        r"""Barebone Fragment Generator, generating determinants solely based on 
        their respective Fragment overlap. Does not conserve 0-space structure!"""
        def __init__(self, propagator: dict):
                self.__dict__.update(propagator.__dict__)

        def excitation(self, i: int, size: int) -> (Sequence, Sequence):
                r"""
                :params det: Determinant we want to excite from
                :params nr:  Number of excitations we wish to generate
                
                :returns: array of keys and array of generation probabilities"""
                if size==0:
                        return [], np.array([])

                # 1. Sample which SCF solutions to spawn on: currently uniformly
                # Idea: Potentially exchange this with SCF transition matrix
                scf_spawn = np.random.randint(0, self.params.nr_scf, size=size)
                
                # 2. Prepare MO list for ex_str:
                #    indices will contain the MO indices corresponding to an 
                #    ex_str, conserving the spin structure of the ex_str
                ex_str = self.index_map_rev[i]
                excited_det = self.generate_det(ex_str)
                indices = np.array([np.arange(n) for n in excited_det.n_electrons], dtype=object)
                for i, spin in enumerate(indices):
                        spin[list(ex_str[1][i])] = list(ex_str[2][i])
               
                # 3. Create MO labels corresponding to determinant we want to spawn from
                old_ijk = [[(ex_str[0], j, k) for k in spin] for j, spin in enumerate(indices)]

                # 4. Map these specific MOs to their fragments
                frags = [[self.fragmap_inv[o] for o in spin] for spin in old_ijk]

                # 5. Count how often which fragment is occupied and therefore
                #    which and how many we need to sample 
                sample_rule = [np.unique(spin, return_counts=True) for spin in frags]

                out = []
                p = []
                for s in scf_spawn:
                        pgen = 1./self.params.nr_scf
                                      
                        # 6. Sample MOs according to sample_rule and produce correct pgen
                        mo_sspace, pgen_mo, localized_on_frag = self.sample_mos(scf=s, rule=sample_rule)
                               
                        if not localized_on_frag:
                                continue

                        pgen *= pgen_mo
                        dexcits, excits, allowed = self.collapse(mo_sspace, excited_det.n_electrons)

                        if not allowed: 
                                continue
                                
                        new_str = (s, dexcits, excits)
                        j = self.index_map[new_str]

                        out.append(j)
                        p.append(pgen)
                

                return np.array(out), np.array(p)
         
        
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

        def excite_local(self, ind: np.ndarray, freq: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
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


