#!/usr/bin/env python
#---- author: Moritz Baumgarten ----# 
#Implementation of a nonorthogonal multireference Coupled Cluster
#quantum Monte Carlo (NOMRCCMC) method.
#The Hilbert space for a common NOMRCCMC calculation is generated
#by a subset of possible exitations of the reference determinats
#(generally obtained from SCF metadynamics)
#Based on Thom [2009], and Spencer and Thom [2016]

import numpy as np
from typing import (
        Tuple, 
        Sequence, 
        NamedTuple
)
from sympy.combinatorics.permutations import Permutation
from functools import reduce

#ALGORITHM (orthogonal):

#       1. Sample Cluster size (decaying with 1/2 exp)
#       2. Sample Excitors from list of simple Excitors
#       3. Collapse sampled Excitors to Cluster
#       4. Do population dynamics, i.e.:
#               4a. Spawn: Per Cluster attempt a single spawn
#               4b. Death
#               4c. Annihilation
#          ==> if Cluster is not in list of simple Excitors we 
#              neglect all excips that died on it
#
#       Additional: Sample projected energy

#class Excitation(NamedTuple):
#        r"""Encodes excitation with respect to a certain SCF solution."""
#        scf: int
#        dex_a: Sequence[int]
#        dex_b: Sequence[int]
#        ex_a: Sequence[int]
#        ex_b: Sequence[int]

class Excitation():
        #reference: SingleDeterminant
        ex_str: Tuple[int, Tuple[Tuple[int], Tuple[int]], Tuple[Tuple[int], Tuple[int]]]

        def indexrep():
                pass

        def __add__():
                pass

class Excitor(): #Again, this class is not necessary as we can store everything in an array of integers 
        r"""Normal ordered walkers comprised of their excitation and their 
        amplitude."""
        def __init__(self, excitation: Sequence, 
                     excips: int) -> None:
                r"""Each excitation has the form of 
                (scf_sol, ((occ_a),(occ_b)), ((vir_a),(vir_b)))

                :params sign:   True if positive sign, False else"""
                self.excitation = excitation
                self.excips     = excips                


class Cluster():        #TODO: ADJUST FOR excitor == []
        r"""Concatenation of Excitors."""
        def __init__(self, excitors: Sequence[Excitor]) -> None:
                self.excitors = excitors
                self.scf = self.excitors[0].excitation[0]
                assert(                                                            
                        np.array(                                                  
                            [self.scf == ex.excitation[0] 
                            for ex in self.excitors]                               
                        ).all()                                                    
                ) #we don't want to excite between different SCF solutions
                self.size = None 
                self.p = None

        def collapse(self):        
                r"""Collapses a sequence of Excitors on 
                the same SCF solution."""
                        
                flat = [flatten(
                            [ex.excitation[i][j] for ex in self.excitors]
                        ) 
                        for i in range(1,3) 
                        for j in range(2)
                       ] # returns list of (sorted, parity)
                
                if not self.check(flat):
                        return None, False
                
                sort = [self.sort(f) for f in flat]
                new = tuple(tuple(c[0]) for c in sort)
                self.excitation = (self.scf, new[:2], new[2:])

                #self.sign = np.prod([s[1] for s in sort]) works with old sort
                self.sign = reduce(lambda i, j: i^j, [s[1] for s in sort])
                if not self.sign: self.sign = 1
                else: self.sign = -1

                #this captures signs from excips already
                self.amplitude = np.prod([ex.excips for ex in self.excitors])

                return self.excitation, self.sign

        def check(self, flat) -> bool:
                r"""Returns the truth value of a collapsed sequence 
                of excitors being 0 or not. Checks whether 
                duplicates are present."""
                return all([len(np.unique(f)) == len(f) for f in flat])

        def sort(self, sequence: Sequence):
                r"""Sorts a sequence and evaluates the parity of the 
                corresponding permutation."""
                index = np.argsort(sequence)
                s = sequence[index]
                p = Permutation(index).is_even
                return s, p

        def is_in_hilbert(self) -> bool: #as long as we are using an indexmap 
                                         #we dont need that function
                r"""Checks whether Cluster is able to carry excips.
                Required evaluation of self.collapse() prior."""
                flat = flatten(self.cluster[1])
                self.in_hilbert = len(flat) == self.excitation_level
                return self.in_hilbert
        

def flatten(seq: Sequence) -> Sequence:
        return np.array([s for sub in seq for s in sub])

if __name__ == '__main__':
        ex1 = Excitor(excitation=[0, [[0,3],[1,2]], [[6,8],[9,10]]], excips=0)
        ex2 = Excitor(excitation=[0, [[2],[0,3]], [[5],[6,11]]], excips=0)
        
        ex1 = Excitor(excitation=(0, ((0,3),(1,2)), ((6,8),(9,10))), excips=0)
        ex2 = Excitor(excitation=(0, ((2,),(0,3)), ((5,),(6,11))), excips=0)
        cluster = Cluster(excitors=[ex1, ex2])
        collapsed = cluster.collapse()
        print(collapsed)

        print(cluster.is_even([1,3,5,4]))
