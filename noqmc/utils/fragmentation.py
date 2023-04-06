import numpy as np


class Fragment():
        r""""""
        def __init__(self, sao):
                r""""""
                self.sao = sao

def fragment(sao: np.ndarray, thresh: float=0.5):
        r"""Taken from https://stackoverflow.com/questions/27016803/find-sets-of-disjoint-sets-from-a-list-of-tuples-or-sets-in-python
        """
        touching_pairs = [set([i for i,r in enumerate(row) if r > thresh]) for row in sao]
        print(touching_pairs)

        out = []
        while len(touching_pairs)>0:
                first, *rest = touching_pairs
                #first = set(first)

                lf = -1
                while len(first)>lf:
                        lf = len(first)

                        rest2 = []
                        for r in rest:
                                if len(first.intersection(set(r)))>0:
                                        first |= set(r)
                                else:
                                        rest2.append(r)     
                        rest = rest2

                out.append(first)
                touching_pairs = rest
        return out



if __name__ == '__main__':
        #generate random hermitean matrix
        sao_rand = np.random.random((9,9)) / 3
        sao_rand += sao_rand.T
        np.fill_diagonal(sao_rand, 1.)
        print(sao_rand)

        #get connectivity
        print(fragment(sao_rand))
