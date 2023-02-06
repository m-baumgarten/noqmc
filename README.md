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

## Usage
To perform a simple NOCI-QMC calculation:

```python
import noqmc.nomrciqmc as noci
from pyscf import gto

mol = gto.M(
    atom = [['H', 0, 0, 0],['H', 0, 0, 1.3]],
    basis = 'sto-3g', unit = 'Angstrom',
)

my_noci = no.NOCIQMC(mol)
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



