import unittest

from .sparse import Sparse, RowSparse, SimpleRowSparse, ZeroOneRowSparse


class TestDecoderRNN(unittest.TestCase):

    def test_add(self):
        """Add concatenates sparses length wise
        """
        s = Sparse(size=[5, 5], row_indices=[0, 1], col_indices=[3,4], values=[1,1])
        t = RowSparse([2, 3], [1,1], length=5)
        res = s + t
        self.assertEqual(res.size[0], 5)
        self.assertEqual(res.size[1], 10)
        self.assertEqual(len(res.values), 4)

    def test_row_sparse(self):
        r = RowSparse([1, 2, 3], [1, 1, 1], 10)
        self.assertEqual(r.size[0], 1)
        self.assertEqual(r.size[1], 10)
        self.assertEqual(len(r.values), 3)

    def test_simple_row_sparse(self):
        s = SimpleRowSparse([1,2,3,4,5])
        # should only have 1 row
        self.assertEqual(s.size[0], 1)
        # length should be 5
        self.assertEqual(s.size[1], 5)
        # should have 5 values
        self.assertEqual(len(s.values), 5)

    def test_zero_one_row_sparse(self):
        z = ZeroOneRowSparse([1,2,3], 100)
        self.assertEqual(z.size[0], 1)
        self.assertEqual(z.size[1], 100)
        self.assertEqual(len(z.values), 3)
        # make sure the values are given the value of 1
        for v in z.values:
            self.assertEqual(v, 1)


if __name__ == '__main__':
    unittest.main()
