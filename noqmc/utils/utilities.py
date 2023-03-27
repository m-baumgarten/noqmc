import os
import sys
import shutil
from typing import Sequence

class Parser():
        r"""Crude Input parser, reading arguments for a nomrci-qmc
        calculation from an input file"""
        def __init__(self, arguments: dict = None) -> None:
                if arguments is not None: self.arguments = arguments
                else: self.arguments = {}

        def parse(self, filename: str) -> dict:
                r"""Extracts raw data from input file."""
                with open(filename, 'r') as f:
                        lines = f.readlines()
                #use partition('#') and rstrip here instead
                lines = [line.replace('\n', '') for line in lines if not (line.startswith('#') or line == '\n')]
                lines = [line.split(' ') for line in lines]
                for line in lines:
                        while '' in line:
                                line.remove('')

                new_lines = []
                for line in lines:
                        columns = []
                        for column in line:
                                if '.' in column:
                                        c = float(column)
                                elif column[0].isdigit():
                                        c = int(column)
                                else: c = column
                                columns.append(c)
                        new_lines.append(columns)
                lines = new_lines
                
                self.compile(lines)
                return self.arguments

        def compile(self, lines: Sequence[str]) -> None: 
                r"""Compiles raw data into dictionary form readable
                by the QMC code."""
                self.arguments = {line[0]: line[1] for line in lines}
                

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
                
