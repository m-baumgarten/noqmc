from qcmagic.auxiliary.qcmagic_standards import ZERO_TOLERANCE
from noqmc.utils.utilities import (
        Parameters,
        Thresholds,
)
import numpy as np

DEFAULT_CCMC_ARGS = Parameters(
    mode='noci',
    verbosity=1,
    seed=69420,
    dt=0.01,
    nr_w=3000,
    A=10,
    c=0.01,
    it_nr=50000,
    delay=20000,
    theory_level=1,
    benchmark=1,
    sampling='uniform'
)

DEFAULT_CIQMC_ARGS = Parameters(
    mode='noci',
    verbosity=1,
    seed=69420,
    dt=0.01,
    nr_w=3000,
    A=10,
    c=0.01,
    it_nr=50000,
    delay=20000,
    theory_level=1,
    benchmark=1,
    sampling='uniform'
)

DEFAULT_DETERMINISTIC_ARGS = Parameters(
    mode='ref',
    verbosity=1,
    seed=69420,
    dt=0.01,
    nr_w=3000,
    it_nr=100,
    theory_level=2,
    benchmark=1,
    localization=0,
    scf_sols=[2,0,0],
    sampling='uniform'
)

THRESHOLDS = Thresholds(
    ov_zero_thresh=5e-06,
    rounding=int(-np.log10(ZERO_TOLERANCE))-12,
    subspace=1e-02
)


