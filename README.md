# noqmc
Collection of nonorthogonal CI-QMC methods. Currently, nonorthogonal truncated CI-QMC and nonorthogonal truncated CCMC are available.

## Installation
Clone the repository into a repository on your path or pythonpath.

```bash
$ git clone https://github.com/m-baumgarten/noqmc.git
```

That's it.

## Dependencies
In order to run noqmc properly, the following python modules are required:  
revqcmagic  
pyblock  
pyscf  


## Usage
To perform a simple NOCI-QMC calculation:

````python
import noqmc.nomrciqmc as no
from pyscf import gto

mol = gto.M(
    atom = [['H', 0, 0, 0],['H', 0, 0, 1.3]],
    basis = 'sto-3g', unit = 'Angstrom',
)

my_no = no.NOCIQMC(mol)
my_no.run()
```


