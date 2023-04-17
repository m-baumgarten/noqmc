import os
import sys
import shutil
import logging
from typing import Sequence
from dataclasses import dataclass, fields

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class Parameters:
        r"""Dataclass to store system parameters."""
        mode: str = None
        verbosity: int = None
        seed: int = None
        dt: float = None
        nr_w: int = None
        A: int = None
        c: float = None
        it_nr: int = None
        delay: int = None
        theory_level: int = None
        benchmark: int = None
        localization: int = None
        scf_sols: list = None
        sampling: str = None
        binning: int = None
        dim: int = None
        nr_scf: int = None
        workdir: str = None
        baseS: str = None

@dataclass
class Thresholds:
        r"""Collection of thresholds used throughout the iteration."""
        ov_zero_thresh: float = None
        rounding: float = None
        subspace: float = None

class Parser():
        r"""Crude Input parser, reading arguments for a nomrci-qmc
        calculation from an input file"""
        def __init__(self, parameters: Parameters=None) -> None:
                if parameters is not None: self.parameters = parameters
                else: self.parameters = Parameters()

        def parse(self, filename: str) -> dict:
                r"""Extracts raw data from input file."""
                with open(filename, 'r') as f:
                        lines = f.readlines()
                lines = [line.rstrip().partition('#') for line in lines]
                lines = [line[0].split() for line in lines if line[0]!='']
                
                new_lines = []
                for line in lines:
                        if len(line[1:]) > 1:
                                new = [self.adjust_type(l) for l in line[1:]]
                        else:
                                new = self.adjust_type(line[1])                                
                        
                        new_lines.append([line[0], new])
                
                self.compile(new_lines)
                return self.parameters

        def adjust_type(self, val: str):
                try: val = int(val)                        
                except ValueError:
                        try: val = float(val)
                        except ValueError: pass
                return val

        def compile(self, lines: Sequence[str]) -> None: 
                r"""Compiles raw data into dictionary form readable
                by the QMC code."""
                pfield = fields(Parameters)
                names = [var.name for var in pfield]
                for line in lines:
                        for var in pfield:
                                if var.name != line[0]:
                                        continue
                                assert(type(line[1]) is var.type)
                        if line[0] not in names:
                                logger.warning(f'You are trying to parse an invalid key: {line[0]}')                                
                        setattr(self.parameters, line[0], line[1])

def setup_workdir(workdir = None):
        if workdir is None:
                if 'output' in os.listdir():
                        shutil.rmtree('output')
                os.mkdir('output')
        else:
                if workdir in os.listdir():
                        shutil.rmtree(workdir)
                os.mkdir(workdir)

if __name__ == '__main__':
        parser = Parser()
        filename = sys.argv[1]
        parser.parse(filename)
        print(parser.parameters)                
