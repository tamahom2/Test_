"""
Tensor Train (TT) implementation using only NumPy, SciPy, and opt_einsum.

This module implements the Tensor Train format and related operations for tensor
completion with adaptive rank.
"""

import numpy as np
import scipy.linalg
import opt_einsum
from copy import copy, deepcopy


def random_normal(shape, scale=1.0):
    """
    Generate random normal tensor of given shape.
    
    Parameters
    ----------
    shape : tuple
        Shape of tensor to generate
    scale : float, optional
        Scale parameter for normal distribution
        
    Returns
    -------
    numpy.ndarray
        Tensor with random normal entries
    """
    return scale * np.random.randn(*shape)


def trim_ranks(dims, tt_rank):
    """
    Trim ranks to ensure they don't exceed theoretical maximum.
    
    Parameters
    ----------
    dims : tuple
        Dimensions of the tensor
    tt_rank : tuple
        TT-ranks to trim
        
    Returns
    -------
    tuple
        Trimmed TT-ranks
    """
    d = len(dims)
    rank_bounds = []
    rank_bounds_rev = []
    
    # Forward cumulative products
    prod = 1
    for i in range(d-1):
        prod *= dims[i]
        rank_bounds.append(min(prod, tt_rank[i]))
    
    # Backward cumulative products
    prod = 1
    for i in range(d-1, 0, -1):
        prod *= dims[i]
        rank_bounds_rev.append(min(prod, tt_rank[i-1]))
    
    rank_bounds_rev.reverse()
    
    return tuple(min(r1, r2) for r1, r2 in zip(rank_bounds, rank_bounds_rev))


def random_isometry(shape):
    """
    Generate random isometric matrix.
    
    Parameters
    ----------
    shape : tuple
        Shape of the isometry (m, n) where m >= n
        
    Returns
    -------
    numpy.ndarray
        Random isometric matrix of shape (m, n)
    """
    m, n = shape
    if m < n:
        raise ValueError("First dimension must be >= second dimension")
    
    Q, _ = np.linalg.qr(np.random.randn(m, n))
    return Q


class TensorTrain:
    """
    Implementation of tensor train (TT) format.
    
    Parameters
    ----------
    cores : list<order 3 tensors>
        List of the TT cores. First core should have shape (1,d0,r0), last has
        shape (r(n-1),dn,1). Last dimension should match first dimension of
        consecutive tensors in the list.
    mode : str or int (default: 'l')
        The orthogonalization mode. Can be an int, or the string 'l' or 'r',
        which respectively get converted to `self.order-1` and `0`
    is_orth : bool (default: False)
        Whether or not the cores are already orthogonalized. If False, the
        cores are orthogonalized during init.
    
    Attributes
    ----------
    cores : list<order 3 tensors>
        List of all the tensor train cores.
    order : int
        The order of the tensor train (number of cores)
    dims : tuple<int> of length self.order
        The outer dimensions of the tensor train, i.e. the shape of the dense
        tensor represented by the tensor train, or `shape[1]` for each core.
    tt_rank : tuple<int> of length self.order-1
        The tt-rank. This `shape[0]` or `shape[2]` for each core. This tuple
        starts with `cores[0].shape[2]`, and hence does always start/end with 1.
    """
    
    def __init__(self, cores, mode="l", is_orth=False):
        self.cores = cores
        self.order = len(cores)
        self.dims = tuple(c.shape[1] for c in cores)
        self.tt_rank = tuple(c.shape[0] for c in cores[1:])
        
        if mode == "l":
            self.mode = self.order - 1
        elif mode == "r":
            self.mode = 0
        else:
            self.mode = mode
            
        if mode is not None:
            if not is_orth:
                # Orthogonalize. If inplace it modifies the argument cores
                self.cores = self._orth_cores(mode, inplace=False)
            self.is_orth = True
        else:
            self.is_orth = False
    
    def orthogonalize(self, mode="l", inplace=True, force_rank=True):
        """
        Orthogonalize the cores with respect to `mode`.
        
        Parameters
        ----------
        mode : int or str (default: "l")
            Orthogonalization mode. If "l", defaults to right-most core, if "r"
            to left-most core.
        inplace : bool (default: True)
            If True then cores are changed in place and return None.
            Otherwise return a TensorTrain object with orthogonalized cores.
        force_rank : bool (default: True)
            If True, check after each step that the rank of the TT hasn't
            lowered. If it has, artificially increase it back by multiplying by
            random isometry.
            
        Returns
        -------
        TensorTrain or None
            If inplace=False, returns new TensorTrain. Otherwise, None.
        """
        new_cores = self._orth_cores(mode=mode, inplace=inplace, force_rank=force_rank)
        
        if not inplace:
            return TensorTrain(new_cores, mode=mode, is_orth=True)
        else:
            self.mode = mode if isinstance(mode, int) else (self.order - 1 if mode == "l" else 0)
            self.is_orth = True
            self.tt_rank = tuple(c.shape[0] for c in self.cores[1:])
    
    def _orth_cores(self, mode="l", inplace=True, force_rank=True):
        """
        Orthogonalize with respect to mode `mode` and return list of new TT cores.
        
        See `orthogonalize` for parameters.
        
        Returns
        -------
        list
            List of orthogonalized cores
        """
        if mode == "l":
            mu = self.order - 1
        elif mode == "r":
            mu = 0
        else:
            if not isinstance(mode, int):
                raise ValueError("Orthogonalization mode should be 'l','r' or int")
            mu = mode
        
        new_cores = [None] * self.order
        if inplace:
            new_cores[mu] = self.cores[mu]
        else:
            new_cores[mu] = deepcopy(self.cores[mu])
        
        # Orthogonalize to left of mu
        if mu > 0:
            for i in range(mu):
                if inplace:
                    C = self.cores[i]
                else:
                    C = deepcopy(self.cores[i])
                
                shape = C.shape
                if i > 0:
                    C = np.reshape(C, (shape[0], shape[1] * shape[2]))
                    C = R @ C
                    K = R.shape[0]
                else:
                    K = 1
                
                C = np.reshape(C, (K * shape[1], shape[2]))
                Q, R = np.linalg.qr(C)
                
                if force_rank and R.shape[0] < R.shape[1]:  # detect rank decrease
                    isometry = random_isometry((R.shape[1], R.shape[0]))
                    R = isometry @ R
                    Q = Q @ isometry.T
                
                new_cores[i] = np.reshape(Q, (K, shape[1], Q.shape[1]))
            
            if mu == self.order - 1:
                C = new_cores[mu]
                C = np.reshape(C, C.shape[:-1])
                C = R @ C
                Q, R = np.linalg.qr(C)
                new_cores[mu - 1] = opt_einsum.contract("ijk,kl->ijl", new_cores[mu - 1], Q)
                new_cores[mu] = np.reshape(R, R.shape + (1,))
            
            if mu < self.order - 1:
                C = new_cores[mu]
                shape = C.shape
                C = np.reshape(C, (shape[0], shape[1] * shape[2]))
                C = R @ C
                C = np.reshape(C, (R.shape[0], shape[1], shape[2]))
                new_cores[mu] = C
        
        # Orthogonalize to the right of mu
        if mu < self.order - 1:
            for i in range(self.order - 1, mu, -1):
                if inplace:
                    C = self.cores[i]
                else:
                    C = deepcopy(self.cores[i])
                
                shape = C.shape
                if i < self.order - 1:
                    C = np.reshape(C, (shape[0] * shape[1], shape[2]))
                    C = C @ R.T
                    K = R.shape[0]
                else:
                    K = 1
                
                C = np.reshape(C, (shape[0], shape[1] * K))
                Q, R = np.linalg.qr(C.T)
                
                if force_rank and R.shape[0] < R.shape[1]:  # detect rank decrease
                    isometry = random_isometry((R.shape[1], R.shape[0]))
                    R = isometry @ R
                    Q = Q @ isometry.T
                
                if i == 0:
                    new_cores[i + 1] = opt_einsum.contract("ji,jkl->ikl", Q, new_cores[i + 1])
                    new_cores[i] = np.reshape(R.T, (1,) + R.shape)
                else:
                    new_cores[i] = np.reshape(Q.T, (Q.shape[1], shape[1], K))
            
            if mu == 0:
                C = new_cores[mu]
                C = np.reshape(C, C.shape[1:])
                C = R @ C.T
                Q, R = np.linalg.qr(C)
                new_cores[mu + 1] = opt_einsum.contract("ji,jkl->ikl", Q, new_cores[mu + 1])
                new_cores[mu] = np.reshape(R.T, (1,) + R.shape[::-1])
            
            if mu > 0:
                C = new_cores[mu]
                shape = C.shape
                C = np.reshape(C, (shape[0] * shape[1], shape[2]))
                C = C @ R.T
                C = np.reshape(C, (shape[0], shape[1], R.shape[0]))
                new_cores[mu] = C
        
        if inplace:
            self.cores = new_cores
            self.is_orth = True
            self.mode = mu
            self.tt_rank = tuple(c.shape[0] for c in new_cores[1:])
        
        return new_cores
    
    @classmethod
    def from_dense(cls, X, eps=1e-14):
        """
        Create TensorTrain from dense tensor using SVD.
        
        Parameters
        ----------
        X : numpy.ndarray
            Dense tensor to decompose
        eps : float, optional
            Truncation tolerance for singular values
            
        Returns
        -------
        TensorTrain
            TT representation of the input tensor
        """
        X = np.asarray(X)
        dims = X.shape
        d = len(dims)
        
        # Initialize ranks
        ranks = [1] * (d + 1)
        cores = []
        
        # Reshape X for first unfolding
        X_mat = X.reshape(dims[0], -1)
        
        # Process from left to right
        for k in range(d-1):
            # SVD and truncate
            U, S, V = np.linalg.svd(X_mat, full_matrices=False)
            
            # Determine rank based on singular values
            tol = eps * np.linalg.norm(S)
            r = np.sum(S > tol)
            ranks[k+1] = r
            
            # Update left-orthogonal core
            cores.append(U[:,:r].reshape(ranks[k], dims[k], ranks[k+1]))
            
            # Prepare matrix for next iteration
            X_mat = np.diag(S[:r]) @ V[:r,:]
            if k < d-2:
                X_mat = X_mat.reshape(r * dims[k+1], -1)
        
        # Last core
        cores.append(X_mat.reshape(ranks[d-1], dims[d-1], ranks[d]))
        
        return cls(cores, is_orth=True)
    
    def dense(self):
        """
        Contract to dense tensor in left-to-right sweep.
        
        Returns
        -------
        numpy.ndarray
            Dense tensor with shape `self.dims`
        """
        C = self.cores[0]
        shape = C.shape
        contracted = np.reshape(C, (shape[0] * shape[1], shape[2]))
        
        for C in self.cores[1:]:
            shape1 = contracted.shape
            shape2 = C.shape
            contracted = contracted @ np.reshape(C, (shape2[0], shape2[1] * shape2[2]))
            contracted = np.reshape(contracted, (shape1[0] * shape2[1], shape2[2]))
        
        contracted = np.reshape(contracted, self.dims)
        return contracted
    
    @classmethod
    def random(cls, dims, tt_rank, mode="l", auto_rank=True):
        """
        Create random TensorTrain of specified shape and rank.
        
        Parameters
        ----------
        dims : iterable of ints
            Dimensions of the tensor
        tt_rank : int or iterable of ints
            If int, all tt-ranks will be the same
        mode : "l", "r" or int
            Orthogonalization mode
        auto_rank : bool (default: True)
            If True, automatically losslessly reduce the rank
            
        Returns
        -------
        TensorTrain
            Random tensor train with specified dimensions and rank
        """
        if isinstance(tt_rank, int):
            tt_rank = [tt_rank] * (len(dims) - 1)
        
        if auto_rank:
            tt_rank = trim_ranks(dims, tt_rank)
        
        ranks = [1] + list(tt_rank) + [1]
        cores = []
        
        for i in range(len(dims)):
            C = random_normal((ranks[i] * dims[i], ranks[i + 1]))
            if i > 0:
                C, _ = np.linalg.qr(C)
            C = np.reshape(C, (ranks[i], dims[i], ranks[i + 1]))
            cores.append(C)
        
        tt = cls(cores, mode=mode, is_orth=True)
        
        if auto_rank:
            tt.orthogonalize(force_rank=False)
        else:
            tt.orthogonalize(force_rank=True)
        
        return tt
    
    def gather(self, idx):
        """
        Gather entries of dense tensor according to indices.
        
        For each row of `idx` this returns one number. This number is obtained
        by multiplying the slices of each core corresponding to each index (in
        a left-to-right fashion).
        
        Parameters
        ----------
        idx : numpy.ndarray
            Array of indices of shape `(N, self.order)` where N is number of entries.
            
        Returns
        -------
        numpy.ndarray
            Result of contraction of shape (N,)
        """
        return self.fast_gather(idx)
    
    def fast_gather(self, idx):
        """
        Faster version of gather for NumPy backend.
        
        Parameters
        ----------
        idx : numpy.ndarray
            Array of indices of shape `(N, self.order)` where N is number of entries.
            
        Returns
        -------
        numpy.ndarray
            Result of contraction of shape (N,)
        """
        idx = idx.T  # Transpose for easier indexing
        N = idx.shape[1]  # Number of entries to gather
        
        # Start with the first core
        result = np.take(self.cores[0].reshape(self.cores[0].shape[1:]), idx[0], axis=0)
        
        # Loop through remaining cores
        for i in range(1, self.order):
            r = self.cores[i].shape[2]
            next_step = np.zeros((N, r))
            
            # For each possible index in this dimension
            for j in range(self.dims[i]):
                # Find which entries have this index
                idx_mask = np.where(idx[i] == j)[0]
                if len(idx_mask) == 0:
                    continue
                
                # Get the core slice and update results
                mat = self.cores[i][:, j, :]
                next_step[idx_mask] = result[idx_mask] @ mat
            
            result = next_step
        
        return result.reshape(-1)
    
    def idx_env(self, alpha, idx, num_cores=1, flatten=True):
        """
        Gather the left and right environment of a TT-core.
        
        Parameters
        ----------
        alpha : int
            Left site of (super)core
        idx : array<int>
            Positions of data to gather
        num_cores : int (default: 1)
            Number of cores in the supercore
        flatten : bool (default: True)
            If True, always flatten result to 2D array. Otherwise result can be
            2D or 3D depending on alpha.
            
        Returns
        -------
        numpy.ndarray
            Environment tensor
        """
        if (alpha + num_cores > self.order) or (alpha < 0):
            raise ValueError("The value of alpha is out of range")
        
        N = len(idx)
        
        # Gather indices of all cores to the left / right of supercore
        left = self.cores[:alpha]
        right = self.cores[alpha + num_cores:]
        
        left_gather = [
            np.take(left[i], idx[:, i], axis=1) for i in range(len(left))
        ]
        
        right_gather = [
            np.take(right[i], idx[:, alpha + num_cores + i], axis=1)
            for i in range(len(right))
        ]
        
        # Contract all the cores to the left / right of the supercore
        if alpha != 0:
            left_env = np.reshape(
                left_gather[0],
                (left_gather[0].shape[1], left_gather[0].shape[2]),
            )
            for M in left_gather[1:]:
                left_env = opt_einsum.contract("ij,jik->ik", left_env, M)
        
        if alpha != len(self.cores) - num_cores:
            right_env = np.reshape(
                right_gather[-1],
                (right_gather[-1].shape[0], right_gather[-1].shape[1]),
            )
            for M in right_gather[-2::-1]:
                right_env = opt_einsum.contract("jik,ki->ji", M, right_env)
        
        # Tensor left and right environments of the supercore
        if alpha == 0:
            env = right_env.T
            if not flatten:
                env = np.reshape(env, (1,) + env.shape)
        elif alpha == len(self.cores) - num_cores:
            env = left_env
            if not flatten:
                env = env.T
                env = np.reshape(env, env.shape + (1,))
        else:
            if flatten:
                env = opt_einsum.contract("bi,jb->bij", left_env, right_env)
                env = np.reshape(env, (N, -1))
            else:
                env = opt_einsum.contract("bi,jb->ibj", left_env, right_env)
        
        return env
    
    def __mul__(self, other):
        if not self.is_orth:
            self._orth_cores()
        new_cores = deepcopy(self.cores)
        new_cores[self.mode] *= other
        return TensorTrain(new_cores, mode=self.mode, is_orth=True)
    
    __rmul__ = __mul__
    
    def __imul__(self, other):
        if not self.is_orth:
            self._orth_cores()
        self.cores[self.mode] *= other
        return self
    
    def __truediv__(self, other):
        return self.__mul__(1 / other)
    
    def __itruediv__(self, other):
        self.__imul__(1 / other)
        return self
    
    def __add__(self, other):
        # Take direct sum for each tensor slice
        new_cores = [np.concatenate((self.cores[0], other.cores[0]), axis=2)]
        
        for C1, C2 in zip(self.cores[1:-1], other.cores[1:-1]):
            r1, d, r2 = C1.shape
            r3, _, r4 = C2.shape
            zeros1 = np.zeros((r1, d, r4))
            zeros2 = np.zeros((r3, d, r2))
            row1 = np.concatenate((C1, zeros1), axis=2)
            row2 = np.concatenate((zeros2, C2), axis=2)
            new_cores.append(np.concatenate((row1, row2), axis=0))
        
        new_cores.append(
            np.concatenate((self.cores[-1], other.cores[-1]), axis=0)
        )
        
        new_tt = TensorTrain(new_cores, is_orth=False)
        new_tt.orthogonalize(force_rank=False)
        return new_tt
    
    def __iadd__(self, other):
        new_tt = self.__add__(other)
        self.cores = new_tt.cores
        self.mode = new_tt.mode
        self.is_orth = new_tt.is_orth
        self.tt_rank = new_tt.tt_rank
        return self
    
    def __neg__(self):
        return (-1) * self
    
    def __sub__(self, other):
        return self + (-1) * other
    
    def __isub__(self, other):
        new_tt = self.__sub__(other)
        self.cores = new_tt.cores
        self.mode = new_tt.mode
        self.is_orth = new_tt.is_orth
        self.tt_rank = new_tt.tt_rank
        return self
    
    def __len__(self):
        return self.order
    
    def __getitem__(self, index):
        return self.cores[index]
    
    def __setitem__(self, index, data):
        self.cores[index] = data
    
    def __repr__(self):
        return (
            f"<TensorTrain of order {self.order} "
            f"with outer dimensions {self.dims}, TT-rank "
            f"{self.tt_rank}, and orthogonalized at mode {self.mode}>"
        )
    
    def round(self, max_rank=None, eps=None, inplace=True):
        """
        Truncate the tensor train.
        
        Parameters
        ----------
        max_rank : int or tuple<int>, optional
            Maximum rank for truncation
        eps : float, optional
            Truncation precision (relative to largest singular value)
        inplace : bool
            If True, round the tensor train in place
            
        Returns
        -------
        TensorTrain
            Truncated tensor train
        """
        if inplace:
            tt = self
        else:
            tt = deepcopy(self)
        
        if max_rank is None and eps is None:
            raise ValueError("At least one of `max_rank` or `eps` has to be specified")
        
        # Scale epsilon relative to norm
        if eps is not None:
            eps *= tt.norm()
        
        if isinstance(max_rank, int):
            max_rank = [max_rank] * tt.order
        
        # Sweep left to right, last core there's nothing to do
        tt.orthogonalize(mode="r")
        for i in range(tt.order - 1):
            # Truncated SVD
            C = tt.cores[i]
            shape = C.shape
            U, S, V = np.linalg.svd(
                np.reshape(C, (shape[0] * shape[1], shape[2])),
                full_matrices=False
            )
            
            if max_rank is not None:
                S = S[: max_rank[i]]
            
            if eps is not None:
                if S[0] > eps:
                    S = S[S > eps]
                else:
                    S = S[:1]
            
            r = len(S)
            tt.cores[i] = np.reshape(U[:, :r], (shape[0], shape[1], r))
            
            # Update next core
            SV = np.diag(S) @ V[:r, :]
            next_core = tt.cores[i + 1]
            next_shape = next_core.shape
            next_core = np.reshape(
                next_core, (next_shape[0], next_shape[1] * next_shape[2])
            )
            next_core = SV @ next_core
            tt.cores[i + 1] = np.reshape(
                next_core, (r, next_shape[1], next_shape[2])
            )
        
        tt.mode = tt.order - 1
        tt.tt_rank = tuple(c.shape[0] for c in tt.cores[1:])
        
        return tt
    
    def increase_rank(self, inc, i=None):
        """
        Increase the rank of the edge between core `i` and `i+1` by `inc`.
        
        Parameters
        ----------
        inc : int
            Amount to increase rank by
        i : int or None
            Position to increase rank at. If None, increase all ranks.
            
        Returns
        -------
        TensorTrain
            Self with increased rank
        """
        # Increase the rank of all nodes
        if i is None:
            for i in range(len(self.tt_rank)):
                self.increase_rank(inc, i)
            return self
        
        # Random isometry of shape (r+inc, r)
        r = self.tt_rank[i]
        A = random_normal((r + inc, r))
        Q, _ = np.linalg.qr(A)
        
        # Apply isometry
        self.cores[i] = opt_einsum.contract("ijk,lk->ijl", self.cores[i], Q)
        self.cores[i + 1] = opt_einsum.contract("ij,jkl->ikl", Q, self.cores[i + 1])
        
        # Update rank information
        new_tt_rank = list(self.tt_rank)
        new_tt_rank[i] += inc
        self.tt_rank = tuple(new_tt_rank)
        
        return self
    
    def sing_vals(self):
        """
        Compute singular values of each unfolding.
        
        Returns
        -------
        list
            List of singular values for each unfolding
        """
        if self.mode != 0:
            self.orthogonalize(mode=0)
        
        sing_vals = []
        
        for i in range(self.order - 1):
            # SVD
            C = self.cores[i]
            shape = C.shape
            U, S, V = np.linalg.svd(
                np.reshape(C, (shape[0] * shape[1], shape[2])),
                full_matrices=False
            )
            sing_vals.append(S)
            self.cores[i] = np.reshape(U, shape)
            
            # Update next core
            SV = np.diag(S) @ V
            next_core = self.cores[i + 1]
            next_shape = next_core.shape
            next_core = np.reshape(
                next_core, (next_shape[0], next_shape[1] * next_shape[2])
            )
            next_core = SV @ next_core
            self.cores[i + 1] = np.reshape(next_core, next_shape)
        
        self.mode = self.order - 1
        self.tt_rank = tuple(c.shape[0] for c in self.cores[1:])
        
        return sing_vals
    
    def rgrad_sparse(self, grad, idx):
        """
        Project sparse euclidean gradient to tangent space.
        
        Parameters
        ----------
        grad : ndarray
            Array containing the values of the sparse gradient
        idx : ndarray of shape (len(grad), self.order)
            Array containing the indices of the dense tensor corresponding to
            the values of the sparse euclidean gradient
            
        Returns
        -------
        TensorTrainTangentVector
            Projected gradient in tangent space
        """
        if self.mode != self.order - 1:
            self.orthogonalize()
        
        right_cores = self._orth_cores(mode="r", inplace=False)
        left_cores = self.cores
        
        for C1, C2 in zip(left_cores, right_cores):
            assert C1.shape == C2.shape, (
                [C.shape for C in left_cores],
                [C.shape for C in right_cores],
            )
        
        N = len(idx)
        
        # Compute left vectors
        left_vectors = [None] * self.order
        for mu in range(self.order - 1):
            left_vectors[mu] = np.zeros((N, self.tt_rank[mu]))
            
            for i in range(self.dims[mu]):
                inds = np.where(idx[:, mu] == i)[0]
                if len(inds) == 0:
                    continue
                
                if mu == 0:
                    update = np.tile(left_cores[0][0, i, :], (len(inds), 1))
                    left_vectors[mu][inds] = update
                else:
                    update = left_vectors[mu - 1][inds] @ left_cores[mu][:, i, :]
                    left_vectors[mu][inds] = update
        
        # Compute right vectors
        right_vectors = [None] * self.order
        for mu in range(self.order - 1, 0, -1):
            right_vectors[mu] = np.zeros((N, self.tt_rank[mu - 1]))
            
            for i in range(self.dims[mu]):
                inds = np.where(idx[:, mu] == i)[0]
                if len(inds) == 0:
                    continue
                
                if mu == self.order - 1:
                    update = np.tile(right_cores[mu][:, i, 0], (len(inds), 1))
                    right_vectors[mu][inds] = update
                else:
                    update = right_vectors[mu + 1][inds] @ right_cores[mu][:, i, :].T
                    right_vectors[mu][inds] = update
        
        # Compute the gradient cores
        grad_cores = [None] * self.order
        
        for mu in range(self.order):
            # Initialize gradient core
            shape = left_cores[mu].shape
            grad_cores[mu] = np.zeros(shape)
            
            # Update for each index in this dimension
            for i in range(self.dims[mu]):
                inds = np.where(idx[:, mu] == i)[0]
                if len(inds) == 0:
                    continue
                
                Z = grad[inds]
                
                if mu > 0:
                    U = left_vectors[mu - 1][inds]
                if mu < self.order - 1:
                    V = right_vectors[mu + 1][inds].T
                
                if mu == self.order - 1:
                    Z = Z.reshape((1, -1))
                    G = (Z @ U).reshape((-1, 1))
                elif mu == 0:
                    Z = Z.reshape((-1, 1))
                    G = (V @ Z).reshape((1, -1))
                else:
                    G = np.einsum("ji,j,kj->ik",U,Z,V)
                
                grad_cores[mu][:, i, :] = G
        
        # Apply gauge conditions to the gradient cores
        self.apply_gradient_gauge_conditions(left_cores, grad_cores)
        
        return TensorTrainTangentVector(grad_cores, left_cores, right_cores)
    
    def apply_gradient_gauge_conditions(self, left_cores, grad_cores):
        """
        Apply gauge conditions to gradient cores inplace.
        
        Parameters
        ----------
        left_cores : list
            List of left-orthogonal cores
        grad_cores : list
            List of gradient cores to update
        """
        for mu in range(self.order - 1):
            r1, r2, r3 = left_cores[mu].shape
            U = np.reshape(left_cores[mu], (r1 * r2, r3))
            
            grad_cores[mu] -= np.reshape(
                U @ (U.T @ np.reshape(grad_cores[mu], (r1 * r2, r3))),
                (r1, r2, r3),
            )
    
    def grad_proj(self, tangent_vector, right_cores=None):
        """
        Project TensorTrainTangentVector to tangent space.
        
        Parameters
        ----------
        tangent_vector : TensorTrainTangentVector
            Tangent vector to project
        right_cores : list, optional
            Right-orthogonal cores
            
        Returns
        -------
        TensorTrainTangentVector
            Projected tangent vector
        """
        return self.tt_proj(
            tangent_vector.to_tt(round=False), right_cores=right_cores
        )
    
    def tt_proj(self, tt, right_cores=None, proj_U=True):
        """
        Project a TT to tangent space of self.
        
        Parameters
        ----------
        tt : TensorTrain
            Tensor train to project
        right_cores : list, optional
            Right-orthogonal cores
        proj_U : bool
            Whether to project onto U
            
        Returns
        -------
        TensorTrainTangentVector
            Projected tangent vector
        """
        if self.mode != self.order - 1:
            self.orthogonalize()
        
        left_cores = self.cores
        if right_cores is None:
            right_cores = self._orth_cores(mode="r", inplace=False)
        
        # List of partial contractions with left/right-orthogonal cores and tt
        right = contract_cores(right_cores, tt, "RL", store_parts=True)
        left = contract_cores(left_cores, tt, "LR", store_parts=True)
        
        # Project grad cores using left and right environments
        grad_cores = []
        grad_cores.append(opt_einsum.contract("ijk,bk->ijb", tt[0], right[1]))
        
        for i in range(1, self.order - 1):
            grad_cores.append(
                opt_einsum.contract(
                    "ai,ijk,bk->ajb",
                    left[i - 1],
                    tt[i],
                    right[i + 1],
                )
            )
        
        grad_cores.append(opt_einsum.contract("ai,ijk->ajk", left[-2], tt[-1]))
        
        # Apply gauge conditions
        if proj_U:
            self.apply_gradient_gauge_conditions(left_cores, grad_cores)
        
        return TensorTrainTangentVector(grad_cores, left_cores, right_cores)
    
    def apply_grad(self, tangent_vector, alpha=1.0, round=True, inplace=False):
        """
        Compute retract of tangent vector.
        
        Parameters
        ----------
        tangent_vector : TensorTrainTangentVector
            Tangent vector, assumed to lie in tangent space at current point
        alpha : float (default: 1.0)
            Stepsize of retract
        round : bool (default: True)
            Whether to round the result to original rank
        inplace : bool (default: False)
            If false, return a new TensorTrain, otherwise update this TT inplace
            
        Returns
        -------
        TensorTrain
            Updated tensor train
        """
        left_cores = tangent_vector.left_cores
        right_cores = tangent_vector.right_cores
        grad_cores = tangent_vector.grad_cores
        
        # Formula from Steinlechner "Riemannian Optimization for High-Dimensional
        # Tensor Completion" (published version), end of page 10
        new_cores = [
            np.concatenate((alpha * grad_cores[0], left_cores[0]), axis=2)
        ]
        
        for U, V, dU in zip(
            left_cores[1:-1], right_cores[1:-1], grad_cores[1:-1]
        ):
            first_row = np.concatenate((V, np.zeros_like(V)), axis=2)
            second_row = np.concatenate((alpha * dU, U), axis=2)
            new_cores.append(np.concatenate((first_row, second_row), axis=0))
        
        new_cores.append(
            np.concatenate(
                (right_cores[-1], left_cores[-1] + alpha * grad_cores[-1]),
                axis=0,
            )
        )
        
        new_tt = TensorTrain(new_cores)
        if round:
            new_tt.round(max_rank=self.tt_rank)
        
        if inplace:
            self.cores = new_tt.cores
            self.tt_rank = new_tt.tt_rank
            self.mode = new_tt.mode
            self.is_orth = new_tt.is_orth
            return self
        else:
            return new_tt
    
    def norm(self):
        """
        Compute Frobenius norm of the tensor train.
        
        Returns
        -------
        float
            Frobenius norm
        """
        if not self.is_orth:
            self.orthogonalize()
        return np.linalg.norm(self.cores[self.mode])
    
    def dot(self, other):
        """
        Compute dot product with other TT with same outer dimensions.
        
        Parameters
        ----------
        other : TensorTrain
            Other tensor train
            
        Returns
        -------
        float
            Dot product
        """
        return contract_cores(self.cores, other.cores)
    
    def __matmul__(self, other):
        return self.dot(other)
    
    def copy(self, deep=True):
        """
        Create a copy of the tensor train.
        
        Parameters
        ----------
        deep : bool
            Whether to create a deep copy
            
        Returns
        -------
        TensorTrain
            Copy of this tensor train
        """
        if deep:
            return deepcopy(self)
        else:
            return copy(self)
    
    def num_params(self):
        """
        Count the number of parameters in the tensor train.
        
        Returns
        -------
        int
            Number of parameters
        """
        return sum([np.prod(C.shape) for C in self.cores])


class TensorTrainTangentVector:
    """
    Class for storing a tangent vector to TT manifold.
    
    A tangent vector at a point on the TT-manifold is a list of cores with the
    same shape as the TT, satisfying a certain gauge condition. This class
    stores both the left-orthogonal and right orthogonal cores of the original
    TT, which are needed for most computations.
    
    Parameters
    ----------
    grad_cores : list<order 3 tensors>
        List of first-order variation cores
    left_cores : list<order 3 tensors>
        List of left-orthogonal cores
    right_cores : list<order 3 tensors>
        List of right-orthogonal cores
    """
    
    def __init__(self, grad_cores, left_cores, right_cores):
        self.grad_cores = grad_cores
        self.left_cores = left_cores
        self.right_cores = right_cores
        self.order = len(grad_cores)
        self.tt_rank = tuple(c.shape[0] for c in grad_cores[1:])
        self.dims = tuple(c.shape[1] for c in grad_cores)
    
    def inner(self, other):
        """
        Compute inner product between two tangent vectors.
        
        Parameters
        ----------
        other : TensorTrainTangentVector
            Other tangent vector
            
        Returns
        -------
        float
            Inner product
        """
        result = 0.0
        for core1, core2 in zip(self.grad_cores, other.grad_cores):
            result += np.dot(core1.reshape(-1), core2.reshape(-1))
        return result
    
    def __matmul__(self, other):
        return self.inner(other)
    
    def norm(self):
        """
        Compute norm of tangent vector.
        
        Returns
        -------
        float
            Frobenius norm
        """
        result = 0
        for c in self.grad_cores:
            result += np.linalg.norm(c) ** 2
        return np.sqrt(result)
    
    @classmethod
    def random(cls, left_cores, right_cores):
        """
        Generate random tangent vector with unit norm gradients.
        
        Parameters
        ----------
        left_cores : list
            Left-orthogonal cores
        right_cores : list
            Right-orthogonal cores
            
        Returns
        -------
        TensorTrainTangentVector
            Random tangent vector
        """
        order = len(left_cores)
        grad_cores = []
        
        for i in range(order):
            C = random_normal(left_cores[i].shape)
            C = C / (np.sqrt(order) * np.linalg.norm(C))
            grad_cores.append(C)
        
        # Project to range of U
        for i in range(order - 1):
            Y_mat = grad_cores[i]
            shape = Y_mat.shape
            Y_mat = np.reshape(grad_cores[i], (shape[0] * shape[1], shape[2]))
            U_mat = left_cores[i]
            U_mat = np.reshape(U_mat, (shape[0] * shape[1], shape[2]))
            Y_mat -= U_mat @ (U_mat.T @ Y_mat)
            grad_cores[i] = np.reshape(Y_mat, shape)
        
        return cls(grad_cores, left_cores, right_cores)
    
    def to_tt(self, round=False):
        """
        Convert to TensorTrain.
        
        Parameters
        ----------
        round : bool
            Whether to round to original rank
            
        Returns
        -------
        TensorTrain
            Tensor train representation
        """
        new_cores = [
            np.concatenate((self.grad_cores[0], self.left_cores[0]), axis=2)
        ]
        
        for U, V, dU in zip(
            self.left_cores[1:-1],
            self.right_cores[1:-1],
            self.grad_cores[1:-1],
        ):
            first_row = np.concatenate((V, np.zeros_like(U)), axis=2)
            second_row = np.concatenate((dU, U), axis=2)
            new_cores.append(np.concatenate((first_row, second_row), axis=0))
        
        new_cores.append(
            np.concatenate(
                (self.right_cores[-1], self.grad_cores[-1]),
                axis=0,
            )
        )
        
        new_tt = TensorTrain(new_cores)
        if round:
            new_tt.round(max_rank=self.tt_rank)
        
        return new_tt
    
    def to_eucl(self, idx):
        """
        Return sparse Euclidean gradient with indices `idx`.
        
        Parameters
        ----------
        idx : ndarray
            Indices for sparse tensor
            
        Returns
        -------
        ndarray
            Sparse Euclidean gradient
        """
        return self.to_tt().gather(idx)
    
    def __repr__(self):
        return (
            f"<TensorTrainTangentVector of order {self.order}, "
            f"outer dimensions {self.dims}, and TT-rank "
            f"{self.tt_rank}>"
        )
    
    def __getitem__(self, index):
        return self.grad_cores[index]
    
    def __setitem__(self, index, data):
        self.grad_cores[index] = data
    
    def __mul__(self, other):
        new_grad_cores = [C * other for C in self.grad_cores]
        return TensorTrainTangentVector(
            new_grad_cores, self.left_cores, self.right_cores
        )
    
    __rmul__ = __mul__
    
    def __imul__(self, other):
        self.grad_cores = [C * other for C in self.grad_cores]
        return self
    
    def __truediv__(self, other):
        return self.__mul__(1 / other)
    
    def __itruediv__(self, other):
        self.__imul__(1 / other)
        return self
    
    def __neg__(self):
        return (-1) * self
    
    def __sub__(self, other):
        return self + (-1) * other
    
    def __isub__(self, other):
        self.__iadd__(-other)
        return self
    
    def __add__(self, other):
        new_grad_cores = [
            C1 + C2 for C1, C2 in zip(self.grad_cores, other.grad_cores)
        ]
        return TensorTrainTangentVector(
            new_grad_cores, self.left_cores, self.right_cores
        )
    
    def __iadd__(self, other):
        self.grad_cores = [
            C1 + C2 for C1, C2 in zip(self.grad_cores, other.grad_cores)
        ]
        return self
    
    def __len__(self):
        return self.order
    
    def copy(self, deep=True):
        """
        Create a copy of the tangent vector.
        
        Parameters
        ----------
        deep : bool
            Whether to create a deep copy
            
        Returns
        -------
        TensorTrainTangentVector
            Copy of this tangent vector
        """
        if deep:
            return deepcopy(self)
        else:
            return copy(self)


def contract_cores(
    cores1, cores2, dir="LR", upto=None, store_parts=False, return_float=True
):
    """
    Compute contraction of one list of TT cores with another.
    
    Parameters
    ----------
    cores1 : list
        First list of TT cores
    cores2 : list
        Second list of TT cores
    dir : str (default: 'LR')
        Direction of contraction; LR (left-to-right), RL (right-to-left)
    upto : int, optional
        Contract only up to this mode
    store_parts : bool (default: False)
        If True, return list of all intermediate contractions
    return_float : bool (default: True)
        If True and the result has total dimension 1, then compress result
        down to a float. Ignored if store_parts=True.
        
    Returns
    -------
    float or ndarray or list
        Contraction result
    """
    result_list = []
    
    if dir == "LR":
        result = opt_einsum.contract("ijk,ajc->iakc", cores1[0], cores2[0])
        result = np.reshape(result, (cores1[0].shape[-1], cores2[0].shape[-1]))
        
        for core1, core2 in zip(cores1[1:upto], cores2[1:upto]):
            if store_parts:
                result_list.append(result)
            result = opt_einsum.contract("ij,ika,jkb->ab", result, core1, core2)
        
        total_dim = np.prod(result.shape)
        if return_float and total_dim == 1 and not store_parts:
            result = float(np.reshape(result, (-1,)))
        
        if store_parts:
            result_list.append(result)
            return result_list
        else:
            return result
            
    elif dir == "RL":
        result = opt_einsum.contract("ijk,ajc->iakc", cores1[-1], cores2[-1])
        result = np.reshape(result, (cores1[-1].shape[0], cores2[-1].shape[0]))
        
        for core1, core2 in zip(cores1[-2:upto:-1], cores2[-2:upto:-1]):
            if store_parts:
                result_list.append(result)
            result = opt_einsum.contract("ab,ika,jkb->ij", result, core1, core2)
        
        total_dim = np.prod(result.shape)
        if return_float and total_dim == 1 and not store_parts:
            result = float(np.reshape(result, (-1,)))
        
        if store_parts:
            result_list.append(result)
            return result_list[::-1]  # reverse result list for RL direction
        else:
            return result
    else:
        raise ValueError(f"Unknown direction '{dir}'")
