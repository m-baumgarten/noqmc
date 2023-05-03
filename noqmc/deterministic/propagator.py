#!/usr/bin/env python
#---- author: Moritz Baumgarten ----# 
#Implementation of a nonorthogonal multireference configuration
#interaction quantum Monte Carlo (NOMRCI-QMC) method.
#The Hilbert space for a common NOMRCI-QMC calculation is generated
#by a subset of possible exitations of the reference determinats
#(generally obtained from SCF metadynamics)
#Based on Booth, Thom and Alavi [2009], and Thom and Head-Gordon [2008]

import numpy as np
from noqmc.nomrccmc.system import System

norm = lambda x: x / np.linalg.norm(x, ord = 2) 

class Propagator(System):
        r"""Class for propagation of the wavefunction/walkers in imaginary time."""

        def __init__(self, system: System) -> None:
                r"""Inherits parameters from System object.

		:param system: System object containing information about Hilbert space"""
                self.__dict__.update(system.__dict__)

                self.E_proj = np.empty(self.params.it_nr)
#                self.Nws = np.empty(self.params.it_nr)
                self.coeffs = np.zeros(
                    [self.params.it_nr+1, self.params.dim]
                )
#                self.initial = np.dot(self.overlap, self.initial) #NOTE
                self.coeffs[0,:] = norm(self.initial.copy()) #* 3000
                
                self.coeffs[0,:] = np.random.random(58)
                self.coeffs[0,:] /= np.linalg.norm(self.coeffs[0,:])
                

                self.curr_it = 0

                tmp = np.load('tmp.npy')
                self.Hi = np.dot(tmp, self.H)

        def calculate_operators(self):
                r"""Calculates the Hamiltonian and overlap"""        
                self._calculatematrices()
                self.propagator = np.eye(self.params.dim) - self.params.dt * self.H
                #self.propagator = self.overlap - self.params.dt * self.H

        def update_propagator(self, E):
#                self.propagator = np.eye(self.params.dim) - self.params.dt * (self.H - E*self.overlap)
                self.propagator = np.eye(self.params.dim) - self.params.dt * (self.Hi - E*np.eye(self.params.dim))

        def E(self) -> float:
                r"""Calculates energy estimator at current iteration."""
                
                coeffs = self.coeffs[self.curr_it,:]
                index = np.where(
                    np.abs(coeffs) == np.max(np.abs(coeffs))
                )[0][0] #get index of maximum value 
                E_proj = np.einsum('i,i->', self.H[index,:], coeffs)
                E_proj /= np.einsum('i,i->', self.overlap[index, :], coeffs)
                self.E_proj[self.curr_it] = E_proj
                return E_proj

        def population_dynamics(self) -> None:    #parallelize for each scf solution
                r"""Spawning/Death in one step due to nonorthogonality. 
		Writes changes to current wavefunction to self.sp_coeffs."""

                E_proj = self.E()
                self.update_propagator(E_proj)

                new = np.dot(self.propagator, self.coeffs[self.curr_it, :])
                new = norm(new)
                self.coeffs[self.curr_it+1, :] = new

                ev, _ = np.linalg.eigh(self.propagator)
                print(f'{self.curr_it} E_proj:      ', 
                    self.E_proj[self.curr_it], ev[-1] 
                )

        def run(self) -> None:
                r"""Executes what is essentially the power method.
                """
                self.calculate_operators()                
                for i in range(self.params.it_nr):
                        self.curr_it = i
                        self.population_dynamics()


if __name__ == '__main__':      
        pass



