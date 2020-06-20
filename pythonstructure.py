## @package myLinAlglib
#  This module contains the linear system class

import numpy
import array
from scipy.sparse import *
from scipy.sparse.linalg import spsolve, cg

## Incremental COO type sparse Matrix
class IncrementalCOOMatrix(object):

    ## Constructor
    #  @param shape Shape of matrix (N,M)
    #  @param dtype Numpy data type
    def __init__(self, shape, dtype):

        if dtype is numpy.int32:
            type_flag = 'i'
        elif dtype is numpy.int64:
            type_flag = 'l'
        elif dtype is numpy.float32:
            type_flag = 'f'
        elif dtype is numpy.float64:
            type_flag = 'd'
        else:
            raise Exception('Dtype not supported.')

        self.dtype = dtype
        self.shape = shape

        self.rows = array.array('i')
        self.cols = array.array('i')
        self.data = array.array(type_flag)

    # Append value to matrix
    # @param i Row index
    # @param j Column index
    # @param v Value
    def append(self, i, j, v):

        m, n = self.shape

        if i >= m or j >= n:
            raise Exception('Index out of bounds')

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    # Convert to Scipy COO matrix
    # @return Scipy COO matrix instance
    def tocoo(self):

        rows = numpy.frombuffer(self.rows, dtype=numpy.int32)
        cols = numpy.frombuffer(self.cols, dtype=numpy.int32)
        data = numpy.frombuffer(self.data, dtype=self.dtype)

        return coo_matrix((data, (rows, cols)),
                             shape=self.shape)

    # Length function
    def __len__(self):

        return len(self.data)

## Linear system of equations
class LinearSystem:

    ## Constructor
    #  @param size     Number of degrees of freedom
    #  @param zerocons Indices of constrained degrees of freedom
    def __init__(self, size, zerocons):
        self.__size = size
        self.__rhs = numpy.zeros(size)
        self.__lhs = csr_matrix((size,size))
        self.__cons = numpy.zeros(size, dtype=bool)
        self.__cons[zerocons] = True

    ## Length function
    def __len__(self):
        return self.__size

    ## Add contribution to the linear system
    #  @param vec   Vector to be added to the right-hand-side
    #  @param mat   Matrix to be added to the left-hand-side
    #  @param rdofs Row degrees of freedom to add to
    #  @param cdofs Column degrees of freedom to add to
    def add(self, vec, mat, rdofs, cdofs=None):
        self.add_to_rhs(vec, rdofs)
        self.add_to_lhs(mat, rdofs, cdofs)

    ## Add contribution to the right-hand-side
    #  @param vec   Vector to be added to the right-hand-side
    #  @param rdofs Row degrees of freedom to add to
    def add_to_rhs(self, vec, rdofs):
        self.__rhs[rdofs] += vec

    ## Add contribution to the left-hand-side
    #  @param mat   Matrix to be added to the left-hand-side
    #  @param rdofs Row degrees of freedom to add to
    #  @param cdofs Column degrees of freedom to add to
    def add_to_lhs(self, mat, rdofs, cdofs=None):
        if not cdofs:
            cdofs = rdofs
        self.__lhs[numpy.ix_(rdofs, cdofs)] += mat

    ## Solve the constrained linear system of equations
    #  @return Solution vector
    def solve(self) -> object:
        lhs_free = self.__lhs[numpy.ix_(~self.__cons, ~self.__cons)]
        rhs_free = self.__rhs[~self.__cons]
        sol = numpy.zeros(len(self))
        sol[~self.__cons] = spsolve(lhs_free, rhs_free)
        return sol


## Linear system of equations
class LinearSparseSystem:

    ## Constructor
    #  @param size     Number of degrees of freedom
    #  @param zerocons Indices of constrained degrees of freedom
    def __init__(self, size, zerocons):
        self.__size = size
        self.__rhs = numpy.zeros(size)
        self.__lhscsr = csr_matrix((size, size), dtype=numpy.float64)
        self.__lhs = IncrementalCOOMatrix((size, size), numpy.float64)
        self.__cons = numpy.zeros(size, dtype=bool)
        self.__cons[zerocons] = True

    ## Length function
    def __len__(self):
        return self.__size

    ## Add contribution to the linear system
    #  @param vec   Vector to be added to the right-hand-side
    #  @param mat   Matrix to be added to the left-hand-side
    #  @param rdofs Row degrees of freedom to add to
    #  @param cdofs Column degrees of freedom to add to
    def add(self, vec, mat, rdofs, cdofs=None):
        self.add_to_rhs(vec, rdofs)
        self.add_to_lhs(mat, rdofs, cdofs)

    ## Add contribution to the right-hand-side
    #  @param vec   Vector to be added to the right-hand-side
    #  @param rdofs Row degrees of freedom to add to
    def add_to_rhs(self, vec, rdofs):
        self.__rhs[rdofs] += vec

    ## Add contribution to the left-hand-side
    #  @param mat   Matrix to be added to the left-hand-side
    #  @param rdofs Row degrees of freedom to add to
    #  @param cdofs Column degrees of freedom to add to
    def add_to_lhs(self, mat, rdofs, cdofs=None):
        if not cdofs:
            cdofs = rdofs
        for i,rdof in enumerate(rdofs):
            for j,cdof in enumerate(cdofs):
                self.__lhs.append(rdof, cdof, mat[i,j])

    ## Solve the constrained linear system of equations
    #  @return Solution vector
    def solve(self) -> object:
        print("Converting COO to CSR...")
        self.__lhscsr = self.__lhs.tocoo().tocsr()
        lhs_free = self.__lhscsr[numpy.ix_(~self.__cons, ~self.__cons)]
        rhs_free = self.__rhs[~self.__cons]
        sol = numpy.zeros(len(self))
        x0 = numpy.full(len(self),0.4)[~self.__cons]
        print("Starting CG solver...")
        sol[~self.__cons] = cg(lhs_free, rhs_free,x0)[0]
        return sol