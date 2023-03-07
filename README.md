# noqmc
Collection of nonorthogonal CI-QMC methods. Currently, nonorthogonal truncated CI-QMC and nonorthogonal truncated CCMC are available.

## Installation
Clone the repository into a repository on your `$PATH` or `$PYTHONPATH`.

```bash
$ git clone https://github.com/m-baumgarten/noqmc.git
```

That's it.

## Dependencies
In order to run noqmc properly, the following python modules are required:  
* `RevQCMagic`  
* [`pyblock`](https://github.com/jsspencer/pyblock)  
* [`PySCF`](https://github.com/pyscf/pyscf)  

`RevQCMagic` can currently only be obtained by contacting [**Dr Alex Thom**](https://www.ch.cam.ac.uk/person/ajwt3) for access. 

##Before running a Calculation
In order to allow for a correct interpretation of Löwdin paired overlaps the `RevQCMagic` standards need to be adjusted accordingly.
This can be done manually in `path/to/RevQCMagic/qcmagic/auxiliary/qcmagic_standards.py`. The following constants need to be changed:

```python
ZERO_TOLERANCE = 1e-13
FLOAT_EPS = np.finfo(float).eps
```

## Usage
To perform a simple NOCI-QMC calculation:

```python
import noqmc.nomrciqmc as noci
from pyscf import gto

mol = gto.M(
    atom = [['H', 0, 0, 0],['H', 0, 0, 1.3]],
    basis = 'sto-3g', unit = 'Angstrom',
)

my_noci = noci.NOCIQMC(mol)
my_noci.run()
```

Similarly, to run a NOCCMC calculation:

```python
import noqmc.nomrccmc as nocc
from pyscf import gto

mol = gto.M(
    atom = [['H', 0, 0, 0],['H', 0, 0, 1.3]],
    basis = 'sto-3g', unit = 'Angstrom',
)

my_nocc = nocc.NOCCMC(mol)
my_nocc.run()
```

However, it is advised to study the examples for improved convergence. Especially the NOCCMC method usually needs an accurate, system-specific set of parameters.

## Documentation
A documentation can be generated from the docstrings with sphinx, similarly to the documentation in `RevQCMagic`. To do so, execute

```bash
$ cd docs
```

Now, make the api reference by executing 

```bash
$ make html
```

Finally, the docs can be read from any webbrowser, e.g. 

```bash
$ firefox build/html/index.html
```

