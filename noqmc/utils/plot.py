import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from noqmc.nomrccmc.postprocessor import Postprocessor

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 11})
rc('text', usetex=True)

SUPPORTED_DATA = [
        'E_proj',
        'Ss',
        'data_summary',
        'coeffs',
        'proj_coeff',
        'Nws',
        'nullspace_evol',
        'coeffs_ad',
        'coeffs_det_no0'
]

class Plot():
        r"""Data visualisation class. Takes data from Postprocessor and plots
        it."""
        def __init__(self) -> None:
                self.cols = 3
                self.eigvals = None
                self.max_lines = 5

        def add_data(self, postpr: Postprocessor) -> dict:
                r"""Reads in data from a Postprocessor, which contains information
                on the population dynamics."""
                self.params = postpr.params
                self.__dict__.update(
                        {key: val 
                        for key,val in postpr.__dict__.items() 
                        if 'eig' in key or 'final' in key}
                )
                
                data = {}
                for key, val in postpr.__dict__.items():
                        if key in SUPPORTED_DATA: 
                                data.update({key: val})
                
                self.data = data 
                return data

        def setup_figure(self, data: dict) -> None:
                #self.rows, rest = divmod(len(data), self.cols)
                self.rows = 2
                #if rest >= 0: self.rows += 1
                
                self.fig, self.ax = plt.subplots(self.rows, self.cols, figsize=(14,7.5), dpi = 150)
                plt.subplots_adjust(wspace = 0.4, hspace = 0.4)

        def plot_energy(self, ax) -> None:
                params = self.params
                x_axis = np.arange(params['it_nr'] + 1) * params['dt']

                ax.plot(x_axis, self.data['E_proj'], label=r'$E(\tau)$')
#                ax.plot(x_axis, self.data['Ss'], label=r'$S(\tau)$')
                
                if self.eigvals is not None:
                        e_corr = np.min(self.eigvals)
                        ax.hlines(e_corr, x_axis[-1], 0, color='black', 
                                linestyle = 'dashed', label=r'$E_{NOCI-SD..}$'
                        )
        
                ax.set_ylabel(r'$E / \mathrm{a.u.}$')
                ax.set_xlabel(r'$\tau$')
                ax.legend(frameon=False)

        def plot_coeffs(self, ax1, ax2) -> None:
                key_coeff = 'coeffs_det_no0'
                key_coeff_ad = 'coeffs_ad'
                params = self.params
                x_axis = np.arange(params['it_nr'] + 1) * params['dt']
                
                ##DETERMINANTS
#                for i in range(self.max_lines):
                for i in range(params['dim']):        
                        ax1.plot(x_axis, self.data[key_coeff][:,i], color = f'C{i}', 
                                label=fr'$\langle D_{i}| \Psi \rangle$' if i <= self.max_lines else None
                        )
                        if self.eigvals is not None:
                                ax1.hlines(self.final[i], x_axis[-1], 0, 
                                        color=f'C{i}', linestyle = 'dashed'
                                )
                ax1.set_ylabel(r'Ground State Coeff. $C_i$')
                ax1.set_xlabel(r'$\tau$')
                ax1.legend(frameon=False)
                
                ##ADIABATIC
                if self.eigvals is None: return None
                print(self.data[key_coeff_ad].shape)                       
                for i in range(self.data[key_coeff_ad].shape[1]):
                        ax2.plot(x_axis, self.data[key_coeff_ad][:,i], 
                                 color = f'C{i}', 
                                 label=f'{i}' if i <= self.max_lines else None)

                ax2.set_ylabel('Contrib. to Coeff')
                ax2.set_xlabel(r'$\tau$')
                ax2.legend(frameon=False)
                
        def plot_walkers(self, ax) -> None:
                x_axis = np.arange(self.params['it_nr']) * self.params['dt']
                
                ax.plot(x_axis, self.data['Nws'])
                ax.set_xlabel(r'$\tau$')
                ax.set_ylabel(r'$N_w$')      

        def plot_stderr(self, ax) -> None:
                data = self.data['data_summary']
                x_axis = np.arange(len(data))
                stderr = [block[4] for block in data]
                stderr_err = [block[5] for block in data]
                
                ax.errorbar(x_axis, stderr, yerr = stderr_err)
                ax.set_xlabel(r'Block size log$_2(n)$')
                ax.set_ylabel(r'Standard deviation / $\mathrm{(a.u.)}^2$')

        def plot_shoulder(self, ax) -> None:
                pass

        def strip_data(self) -> None:
                r"""Checks what data was passed to the class and sets it up for
                plotting."""
                pass

        def plot_nullspace(self, ax) -> None:
                x_axis = np.arange(self.params['it_nr'] + 1) * self.params['dt']
                for i in range(self.params['dim']):
                        ax.plot(x_axis,
                                self.data['nullspace_evol'][:,i], color = f'C{i}', 
                                label=f'{i}' if i <= self.max_lines else None
                        )

                ax.set_ylabel('0-space components')
                ax.set_xlabel(r'$\tau$')
                ax.legend(frameon=False)
               

        def plot_data(self) -> None:
                r""""""
                #TODO adjust according to data passed 
                
                self.plot_energy(self.ax[0,1])
                self.plot_coeffs(self.ax[0,0], self.ax[1,0])
                self.plot_walkers(self.ax[0,2])
#                self.plot_stderr(self.ax[1,1])
                self.plot_nullspace(self.ax[1,2])
#                self.ax[1,2].set_axis_off()
                plt.savefig('tmp.png')
                plt.show()
