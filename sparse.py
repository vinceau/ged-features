
class Sparse(object):
    """Wrapper for sparse vectors. Makes joining them together easier.
    """

    def __init__(self, row_indices, col_indices, values, size):
        self.row_indices = row_indices
        self.col_indices = col_indices
        self.values = values
        self.size = size

    def __add__(self, other):
        """Not really add but concatenate the two sparses together
        """
        h, w = self.size
        col_indices = self.col_indices + [w + i for i in other.col_indices]
        row_indices = self.row_indices + other.row_indices
        values = self.values + other.values
        oh, ow = other.size
        size = [max(h, oh), w + ow]
        return Sparse(row_indices, col_indices, values, size)

    def stack(self, other):
        h, w = self.size
        row_indices = self.row_indices + [h + i for i in other.row_indices]
        col_indices = self.col_indices + other.col_indices
        values = self.values + other.values
        oh, ow = other.size
        size = [h + oh, max(w, ow)]
        return Sparse(row_indices, col_indices, values, size)

    def to_torch(self):
        import torch
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
        super(RowSparse, self).__init__(row_indices, col_indices, values, [1, length])


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

