import os
import sys
import time
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
                #Log().info(f'Arguments: {self.arguments}')
                

class Log():
        r"""Crude logger, dumping stuff to a file."""
        def __init__(self, name: str = __name__, filename: str = 'log.out') -> None:
                self.name = name
                self.file = filename 
                if self.file in os.listdir(os.getcwd()):
                        os.remove(os.path.join(os.getcwd(),self.file))
                self.init_time = time.perf_counter()

        def append(self, prefix: str, data: str) -> None:
                with open(self.file, 'a') as f:
                        f.write(f'{prefix}:     {data}\n')

        def info(self, data: str) -> None:
                self.append('info', data)
                
        def warning(self, data: str) -> None:
                self.append('WARNING:', data)


class Timer():
        r"""Create Timer"""
        def __init__(self) -> None:
                self.time = [time.perf_counter()]
                self.curr_t = self.time[0]
                self.func_names = ['Initialization']                            
                self.log = Log(filename = './time.out')

        def lapse(self, func) -> None:
                t = time.perf_counter()
                self.time.append(t)
                self.func_names.append(str(func))
                self.log.info(f'Time for {func}:     {t - self.curr_t}')
                self.curr_t = t

        def stop(self) -> None:
                pass

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
                
