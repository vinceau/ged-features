import numpy as np
from scipy import sparse

class Sparse(object):
    """Wrapper for sparse vectors. Makes joining them together easier.
    """

    def __init__(self, size, row_indices=None, col_indices=None, values=None):
        self.size = size
        self.row_indices = row_indices if row_indices else []
        self.col_indices = col_indices if col_indices else []
        self.values = values if values else []

    def __add__(self, other):
        """Not really add but concatenate the two sparses together
        """
        h, w = self.size
        col_indices = self.col_indices + [w + i for i in other.col_indices]
        row_indices = self.row_indices + other.row_indices
        values = self.values + other.values
        oh, ow = other.size
        size = [max(h, oh), w + ow]
        return Sparse(size, row_indices, col_indices, values)

    def transpose(self):
        size = list(reversed(self.size))
        return Sparse(size, self.col_indices, self.row_indices, self.values)

    def stack(self, other):
        h, w = self.size
        row_indices = self.row_indices + [h + i for i in other.row_indices]
        col_indices = self.col_indices + other.col_indices
        values = self.values + other.values
        oh, ow = other.size
        size = [h + oh, max(w, ow)]
        return Sparse(size, row_indices, col_indices, values)

    def to_numpy(self, as_matrix=False):
        matrix = sparse.coo_matrix((self.values, (self.row_indices, self.col_indices)), self.size, dtype=np.float32)
        if as_matrix:
            return matrix

        return np.asarray(matrix.todense())

    def to_torch(self):
        import torch
        if not self.values:  # we're an empty sparse
            return torch.sparse.FloatTensor(torch.Size(self.size))

        indices = torch.LongTensor([self.row_indices, self.col_indices])
        values = torch.FloatTensor(self.values)
        return torch.sparse.FloatTensor(
                indices, 
                values, 
                torch.Size(self.size)
        )


class RowSparse(Sparse):
    """We only ever have a single row, so the row indices are always zero
    """

    def __init__(self, col_indices, values, length):
        row_indices = [0] * len(col_indices)
        super(RowSparse, self).__init__([1, length], row_indices, col_indices, values)


class SimpleRowSparse(RowSparse):
    """infer column indices and length from values
    """
    def __init__(self, values):
        length = len(values)
        col_indices = list(range(length))
        super(SimpleRowSparse, self).__init__(col_indices, values, length)


class ZeroOneRowSparse(RowSparse):
    """The values are only going to be zero or one.
    Automatically generate a list of ones for each column in col_indices
    """

    def __init__(self, col_indices, length):
        values = [1] * len(col_indices)
        super(ZeroOneRowSparse, self).__init__(col_indices, values, length)

    @staticmethod
    def from_index(index, length):
        return ZeroOneRowSparse([index], length)

