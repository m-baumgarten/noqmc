import numpy as np
#from noqmc.nomrciqmc.propagator import Propagator
from noqmc.nomrccmc.system import System
from noqmc.utils.utilities import Parameters
from typing import Sequence, Tuple

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
                pass

        def double(self):
                pass

        def single(self):
                pass

class HeatBathGenerator(ExcitationGenerator):
        r""""""
        def __init__(self, propagator: dict) -> None:
                H = propagator.H
                overlap = propagator.overlap

                HAMILTONIAN_PRECOMPUTED = not any(np.isnan(H).flatten())
                assert(HAMILTONIAN_PRECOMPUTED)
                OVERLAP_PRECOMPUTED = not any(np.isnan(overlap).flatten())
                assert(OVERLAP_PRECOMPUTED)
                
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
               

class LocalHeatBathGenerator(HeatBathGenerator, OrthogonalGenerator):
        r""""""
        def __init__(self):
                pass

        def excitation(self, i: int, size: int) -> (np.ndarray, np.ndarray):
                pass

class FragmentGenerator(ExcitationGenerator):
        r""""""
        def __init__(self, propagator: dict):
                self.params = propagator.params
                pass


        def excitation(self, i: int, size: int) -> (Sequence, Sequence):
                r"""
                :params det: Determinant we want to excite from
                :params nr:  Number of excitations we wish to generate
                
                :returns: array of keys and array of generation probabilities"""
                ex_str = self.index_map[i]
                # Idea: Potentially exchange this with SCF transition matrix 
                if size==0:
                        return [], np.array([])

                # 1. Sample which SCF solutions to spawn on: currently uniformly
                scf_spawn = np.random.randint(0, self.params.nr_scf, size=size)
                
                # 2. Prepare MO list for ex_str:
                #    indices will contain the MO indices corresponding to an 
                #    ex_str, conserving the spin structure of the ex_str
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
                        j = self.index_map_inv[new_str]

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


