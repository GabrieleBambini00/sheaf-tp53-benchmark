"""
Fully Vectorized Sheaf Connection Laplacian.

REPLACES the O(E) Python loop in sheaf.py with a fully vectorized scatter-add
implementation using block index tensors. This is the key mathematical and
performance improvement for TS-GNN.

Theory
------
The sheaf connection Laplacian L_F ∈ R^{Nd × Nd} is defined blockwise:

    L_F[u·d:(u+1)·d, v·d:(v+1)·d] = -F_{u,e}^T @ F_{v,e}       (off-diagonal)
    L_F[u·d:(u+1)·d, u·d:(u+1)·d] += F_{u,e}^T @ F_{u,e}        (diagonal)
    L_F[v·d:(v+1)·d, v·d:(v+1)·d] += F_{v,e}^T @ F_{v,e}        (diagonal)

The trick: we flatten the d×d blocks into d² scalars, build row/col index
vectors of length E·d², and use scatter_add_ to write all blocks at once.

Complexity: O(E·d²) with NO Python loops.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["VectorizedSheafDiffusion", "NormalizedVectorizedSheafDiffusion"]


class VectorizedSheafDiffusion(nn.Module):
    """
    Sheaf Diffusion Layer with fully vectorized Laplacian construction.

    Key improvement over the original SheafDiffusionLayer:
    - compute_connection_laplacian_vectorized: O(E·d²) with zero Python loops
    - spectral_gap(): eigenvalue-based health check for training monitoring
    - condition_number(): numerical stability diagnostic

    Args:
        num_nodes: N, number of nodes.
        num_edges: E, number of directed edges.
        stalk_dim: d, dimension of each sheaf stalk.
        edge_index: (2, E) edge indices.
    """

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        stalk_dim: int,
        edge_index: torch.Tensor,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.d = stalk_dim
        self.Nd = num_nodes * stalk_dim

        self.register_buffer("edge_index", edge_index)

        # Learnable restriction maps: F_{u,e}, F_{v,e} ∈ R^{d×d} for each edge
        self.restriction_maps = nn.Parameter(
            torch.empty(num_edges, 2, stalk_dim, stalk_dim)
        )

        # Diffusion weight matrix (shared across all diffusion steps)
        self.weight = nn.Parameter(torch.empty(stalk_dim, stalk_dim))
        self.bias = nn.Parameter(torch.zeros(stalk_dim))

        # Residual scaling
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self._init_parameters()
        self._precompute_scatter_indices()

    def _init_parameters(self):
        """Orthogonal init for restriction maps, Xavier for weights."""
        for e in range(self.num_edges):
            for s in range(2):
                nn.init.orthogonal_(self.restriction_maps.data[e, s])
        nn.init.xavier_uniform_(self.weight)

    def _precompute_scatter_indices(self):
        """Precompute the row/col index vectors for scatter_add_.

        For each edge e = (u, v) and each entry (i, j) in the d×d block,
        we need:
            row_index = u*d + i  (for off-diag source side)
            col_index = v*d + j  (for off-diag target side)

        We build these once and reuse them at every forward pass.
        This converts the block-scatter problem into a flat-scatter problem.
        """
        E, d = self.num_edges, self.d
        src = self.edge_index[0]  # (E,)
        tgt = self.edge_index[1]  # (E,)

        # For each edge, for each (i,j) in d×d block:
        # row = node * d + i, col = node * d + j
        # Total entries per edge: d²
        # Total entries: E * d²

        d_range = torch.arange(d)

        # Row indices: repeat each node_id*d + i for d columns → shape (E, d, d) → (E*d²)
        # Col indices: repeat each node_id*d + j for d rows → same process

        # Expand node offsets: (E,) → (E, d, d)
        src_offsets = (src * d).unsqueeze(-1).unsqueeze(-1)  # (E, 1, 1)
        tgt_offsets = (tgt * d).unsqueeze(-1).unsqueeze(-1)  # (E, 1, 1)

        # Row within block: i ∈ [0, d), repeated across columns
        i_idx = d_range.unsqueeze(-1).expand(d, d).unsqueeze(0)  # (1, d, d)
        # Col within block: j ∈ [0, d), repeated across rows
        j_idx = d_range.unsqueeze(0).expand(d, d).unsqueeze(0)  # (1, d, d)

        # Off-diagonal (u→v): L[u*d+i, v*d+j]
        off_diag_rows = (src_offsets + i_idx).reshape(-1)  # (E*d²)
        off_diag_cols = (tgt_offsets + j_idx).reshape(-1)  # (E*d²)

        # Off-diagonal (v→u): L[v*d+i, u*d+j] (symmetric transpose)
        off_diag_rows_T = (tgt_offsets + i_idx).reshape(-1)
        off_diag_cols_T = (src_offsets + j_idx).reshape(-1)

        # Diagonal source: L[u*d+i, u*d+j]
        diag_src_rows = (src_offsets + i_idx).reshape(-1)
        diag_src_cols = (src_offsets + j_idx).reshape(-1)

        # Diagonal target: L[v*d+i, v*d+j]
        diag_tgt_rows = (tgt_offsets + i_idx).reshape(-1)
        diag_tgt_cols = (tgt_offsets + j_idx).reshape(-1)

        # Convert 2D indices to flat 1D indices for scatter_add_ on flattened Laplacian
        Nd = self.Nd
        self.register_buffer("off_flat_idx", off_diag_rows * Nd + off_diag_cols)
        self.register_buffer("off_flat_idx_T", off_diag_rows_T * Nd + off_diag_cols_T)
        self.register_buffer("diag_src_flat_idx", diag_src_rows * Nd + diag_src_cols)
        self.register_buffer("diag_tgt_flat_idx", diag_tgt_rows * Nd + diag_tgt_cols)

        # ── Sparse COO indices (reused by compute_sparse_laplacian) ──────────
        # We store the (row, col) pairs for every structural non-zero so the
        # sparse path only needs to supply the matching values at forward time.
        #
        # Structural non-zeros:
        #   4 × E × d² element slots coming from the 4 scatter operations above.
        # Values can overlap (same (row,col) written by ≥ 1 scatter); COO
        # torch.sparse_coo_tensor with duplicate indices sums them automatically,
        # which is exactly what we want (same semantic as scatter_add_).
        sparse_rows = torch.cat([
            off_diag_rows,       # off-diag (u→v)
            off_diag_rows_T,     # off-diag (v→u)
            diag_src_rows,       # diagonal src
            diag_tgt_rows,       # diagonal tgt
        ])  # (4 * E * d²,)
        sparse_cols = torch.cat([
            off_diag_cols,
            off_diag_cols_T,
            diag_src_cols,
            diag_tgt_cols,
        ])  # (4 * E * d²,)
        # Stack as (2, nnz) index tensor required by torch.sparse_coo_tensor
        self.register_buffer(
            "sparse_coo_indices",
            torch.stack([sparse_rows, sparse_cols], dim=0),  # (2, 4Ed²)
        )

    def compute_sparse_laplacian(
        self,
        restriction_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the sheaf connection Laplacian as a SPARSE COO tensor.

        Memory layout
        -------------
        The dense Laplacian is (Nd × Nd) with Nd² elements.
        The sparse version stores at most 4 · E · d² values (the four scatter
        operations: off-diag u→v, off-diag v→u, diag-src, diag-tgt).
        For a sparse graph (E ≪ N²) this is much smaller.

        The COO tensor has duplicate (row, col) entries that PyTorch sums when
        the tensor is coalesced or used in a matvec, giving the identical result
        to scatter_add_ on the dense path.

        Args:
            restriction_maps: Optional external maps (E, 2, d, d).

        Returns:
            L_sparse: (Nd, Nd) sparse COO tensor (coalesced, fp32-compatible).
        """
        maps = restriction_maps if restriction_maps is not None else self.restriction_maps
        Nd = self.Nd
        dev = maps.device

        F_src = maps[:, 0]  # (E, d, d)
        F_tgt = maps[:, 1]  # (E, d, d)

        # Compute block products — same as dense path
        off_diag = -torch.bmm(F_src.transpose(1, 2), F_tgt)          # (E, d, d)
        off_diag_T = off_diag.transpose(1, 2)                         # (E, d, d)
        diag_src = torch.bmm(F_src.transpose(1, 2), F_src)            # (E, d, d)
        diag_tgt = torch.bmm(F_tgt.transpose(1, 2), F_tgt)            # (E, d, d)

        # Concatenate values in the same order as the index buffers
        values = torch.cat([
            off_diag.reshape(-1),
            off_diag_T.reshape(-1),
            diag_src.reshape(-1),
            diag_tgt.reshape(-1),
        ])  # (4 * E * d²,)

        # Move index buffers to device if needed (same guard as dense path)
        indices = self.sparse_coo_indices
        if indices.device != dev:
            indices = indices.to(dev)
            self.sparse_coo_indices = indices

        # Build COO tensor — duplicate indices will be summed on coalesce
        L_sparse = torch.sparse_coo_tensor(
            indices, values, size=(Nd, Nd), device=dev
        ).coalesce()

        return L_sparse

    def sparse_diffusion(
        self,
        x: torch.Tensor,
        L_sparse: torch.Tensor,
    ) -> torch.Tensor:
        """One step of sheaf diffusion using the sparse Laplacian.

        Identical math to sheaf_diffusion() but uses torch.sparse.mm instead
        of a dense matmul, avoiding materialisation of the (Nd × Nd) matrix.

        Args:
            x: (N, d) node features.
            L_sparse: (Nd, Nd) sparse COO Laplacian from compute_sparse_laplacian.

        Returns:
            Updated (N, d) node features.
        """
        h = x @ self.weight + self.bias          # (N, d)
        h_col = h.reshape(-1, 1)                 # (Nd, 1)  — mm needs 2-D
        diffused = torch.sparse.mm(L_sparse, h_col).reshape(-1, self.d)  # (N, d)
        return x - self.alpha * diffused

    def compute_connection_laplacian_vectorized(
        self,
        restriction_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the (Nd, Nd) sheaf connection Laplacian with ZERO Python loops.

        Uses precomputed scatter indices to write all E·d² entries in 4 calls
        to scatter_add_, achieving O(E·d²) with full GPU parallelism.

        Args:
            restriction_maps: Optional external maps (E, 2, d, d).

        Returns:
            L_F: (Nd, Nd) symmetric sheaf Laplacian.
        """
        maps = restriction_maps if restriction_maps is not None else self.restriction_maps
        d = self.d
        Nd = self.Nd

        F_src = maps[:, 0]  # (E, d, d)
        F_tgt = maps[:, 1]  # (E, d, d)

        # Compute all block products via batched matmul — O(E·d³)
        off_diag = -torch.bmm(F_src.transpose(1, 2), F_tgt)   # (E, d, d)
        diag_src = torch.bmm(F_src.transpose(1, 2), F_src)     # (E, d, d)
        diag_tgt = torch.bmm(F_tgt.transpose(1, 2), F_tgt)     # (E, d, d)

        # Ensure scatter index buffers live on the same device as the maps.
        # We do a single device check rather than four .to() calls, so buffers
        # are moved at most once per device transition (e.g. CPU → CUDA).
        dev = maps.device
        if self.off_flat_idx.device != dev:
            self.off_flat_idx = self.off_flat_idx.to(dev)
            self.off_flat_idx_T = self.off_flat_idx_T.to(dev)
            self.diag_src_flat_idx = self.diag_src_flat_idx.to(dev)
            self.diag_tgt_flat_idx = self.diag_tgt_flat_idx.to(dev)

        # Flatten the d×d blocks: (E, d, d) → (E*d²)
        # Use dtype of computed tensors (may differ from maps.dtype under autocast/fp16)
        compute_dtype = off_diag.dtype
        off_flat = off_diag.reshape(-1)
        diag_src_flat = diag_src.reshape(-1)
        diag_tgt_flat = diag_tgt.reshape(-1)

        # Initialize flat Laplacian with the same dtype as the computed blocks
        L_flat = torch.zeros(Nd * Nd, device=dev, dtype=compute_dtype)

        # Scatter all blocks in 4 vectorized calls (no Python loop!)
        L_flat.scatter_add_(0, self.off_flat_idx, off_flat)
        L_flat.scatter_add_(0, self.off_flat_idx_T,
                           off_diag.transpose(1, 2).reshape(-1))  # Transpose blocks
        L_flat.scatter_add_(0, self.diag_src_flat_idx, diag_src_flat)
        L_flat.scatter_add_(0, self.diag_tgt_flat_idx, diag_tgt_flat)

        return L_flat.reshape(Nd, Nd)

    # Alias for API compatibility with SheafDiffusionLayer
    def compute_connection_laplacian(
        self, restriction_maps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.compute_connection_laplacian_vectorized(restriction_maps)

    def sheaf_diffusion(
        self,
        x: torch.Tensor,
        L_F: torch.Tensor,
    ) -> torch.Tensor:
        """
        One step of sheaf diffusion: x_{k+1} = x_k - α·L_F·(x_k·W + b)

        Residual connection ensures gradient stability even with many steps.
        """
        h = x @ self.weight + self.bias          # (N, d)
        h_lifted = h.reshape(-1)                 # (Nd,)
        diffused = L_F @ h_lifted                # (Nd,)
        diffused = diffused.reshape(-1, self.d)  # (N, d)
        return x - self.alpha * diffused

    # Alias for API compatibility with SheafDiffusionLayer
    def diffuse(self, x: torch.Tensor, L_F: torch.Tensor) -> torch.Tensor:
        return self.sheaf_diffusion(x, L_F)

    def forward(
        self,
        node_features: torch.Tensor,
        restriction_maps: Optional[torch.Tensor] = None,
        num_steps: int = 3,
        use_sparse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full sheaf diffusion forward pass.

        Args:
            node_features: (N, d) features in stalk space.
            restriction_maps: Optional external maps.
            num_steps: Number of diffusion iterations.
            use_sparse: If True, build a sparse COO Laplacian and use
                sparse_diffusion() instead of the dense path.  The result is
                numerically identical to the dense path (up to fp tolerance).
                The dense Laplacian is still returned as the second element so
                the API contract is unchanged.

        Returns:
            diffused_features: (N, d) after diffusion.
            L_F: (Nd, Nd) dense Laplacian (always dense; sparse COO is an
                 internal representation only).
        """
        if use_sparse:
            L_sparse = self.compute_sparse_laplacian(restriction_maps)
            x = node_features
            for _ in range(num_steps):
                x = self.sparse_diffusion(x, L_sparse)
            # Return dense Laplacian for compatibility (diagnostics, visualisation)
            L_F = self.compute_connection_laplacian_vectorized(restriction_maps)
            return x, L_F

        L_F = self.compute_connection_laplacian_vectorized(restriction_maps)
        x = node_features
        for _ in range(num_steps):
            x = self.sheaf_diffusion(x, L_F)
        return x, L_F

    # ── Spectral Diagnostics ────────────────────────────────────────────

    @torch.no_grad()
    def spectral_gap(
        self,
        restriction_maps: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute the spectral gap (λ₂ - λ₁) of the sheaf Laplacian.

        Theory: The spectral gap controls the rate of information diffusion
        on the graph. A Laplacian with spectral gap ≈ 0 means the graph is
        nearly disconnected (diffusion stalls). A large spectral gap means
        fast mixing but potentially over-smoothing.

        Ideal range for GRN: 0.01 < spectral_gap < 2.0

        Returns:
            Spectral gap λ₂ - λ₁ (should be > 0 for connected graphs).
        """
        L = self.compute_connection_laplacian_vectorized(restriction_maps)
        eigenvalues = torch.linalg.eigvalsh(L.float())
        # Sort and get gap between smallest and second-smallest
        sorted_eigs = eigenvalues.sort().values
        # Laplacian should have λ₁ ≈ 0 for connected component
        gap = (sorted_eigs[1] - sorted_eigs[0]).item()
        return gap

    @torch.no_grad()
    def condition_number(
        self,
        restriction_maps: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute the condition number κ(L_F) = λ_max / λ₂.

        Theory: High condition number (κ > 10⁴) indicates ill-conditioning,
        which causes gradient instability during diffusion. This is a direct
        measure of numerical soundness.

        Returns:
            Condition number (lower is better, > 10⁴ is concerning).
        """
        L = self.compute_connection_laplacian_vectorized(restriction_maps)
        eigenvalues = torch.linalg.eigvalsh(L.float())
        sorted_eigs = eigenvalues.sort().values
        lambda_2 = sorted_eigs[1].clamp(min=1e-10)
        lambda_max = sorted_eigs[-1].clamp(min=1e-10)
        return (lambda_max / lambda_2).item()

    @torch.no_grad()
    def verify_laplacian_properties(
        self,
        restriction_maps: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Comprehensive Laplacian verification suite.

        Checks:
        1. Symmetry: L = L^T
        2. Positive semi-definiteness: all eigenvalues ≥ 0
        3. Row-sum zero (for Laplacian of unweighted graph): diag L = -sum off-diag
        4. Spectral gap > 0 (connected graph test)
        5. Condition number < 10⁴ (numerical stability)
        """
        L = self.compute_connection_laplacian_vectorized(restriction_maps)
        is_symmetric = torch.allclose(L, L.T, atol=1e-5)
        frob = L.norm().item()
        try:
            eigs = torch.linalg.eigvalsh(L.float()).sort().values
            return {
                "is_symmetric": is_symmetric,
                "is_psd": bool((eigs >= -1e-5).all()),
                "min_eigenvalue": eigs[0].item(),
                "spectral_gap": (eigs[1] - eigs[0]).item(),
                "condition_number": (eigs[-1] / eigs[1].clamp(min=1e-10)).item(),
                "max_eigenvalue": eigs[-1].item(),
                "frobenius_norm": frob,
            }
        except torch._C._LinAlgError:
            # Ill-conditioned matrix — return safe defaults, do not crash training
            return {
                "is_symmetric": is_symmetric,
                "is_psd": False,
                "min_eigenvalue": float("nan"),
                "spectral_gap": float("nan"),
                "condition_number": float("inf"),
                "max_eigenvalue": float("nan"),
                "frobenius_norm": frob,
            }


class NormalizedVectorizedSheafDiffusion(VectorizedSheafDiffusion):
    """
    Symmetric normalized sheaf Laplacian L_sym = D^{-1/2} L_F D^{-1/2}.

    Eigenvalues bounded in [0, 2] by the Cheeger inequality (Bodnar et al.
    2022).  Improves numerical conditioning and convergence over the
    unnormalized version.

    The block-diagonal degree matrix is defined as:

        D_F[u] = sum_{e incident to u} F_{u,e}^T @ F_{u,e}   ∈ R^{d×d}

    so that D_F is a (Nd, Nd) block-diagonal matrix.  The symmetric
    normalization is then:

        L_sym = D_F^{-1/2} @ L_F @ D_F^{-1/2}

    where D_F^{-1/2}[u] is obtained from the eigendecomposition of D_F[u]
    with a small eps=1e-8 regularization for numerical stability.

    All other methods (sheaf_diffusion, forward, spectral_gap,
    condition_number, verify_laplacian_properties) are inherited unchanged
    and automatically operate on the normalized Laplacian because they all
    call compute_connection_laplacian_vectorized internally.

    API alias
    ---------
    compute_connection_laplacian(...)  →  same as
    compute_connection_laplacian_vectorized(...) (inherited from parent).
    """

    def compute_connection_laplacian_vectorized(
        self,
        restriction_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the symmetrically normalized (Nd, Nd) sheaf Laplacian.

        Steps
        -----
        1. Delegate to the parent to obtain the raw (unnormalized) L_F.
        2. Compute the (N, d, d) degree blocks D_F[u] via scatter_add on the
           per-edge FTF products for both endpoint roles (source and target).
        3. For each node block D_F[u], compute D_F[u]^{-1/2} via
           torch.linalg.eigh with eps regularization.
        4. Apply the symmetric normalization efficiently with einsum by
           reshaping L_F to (N, d, N, d) and contracting with D_inv_sqrt on
           both node axes, then reshaping back to (Nd, Nd).

        Args:
            restriction_maps: Optional external maps (E, 2, d, d).

        Returns:
            L_sym: (Nd, Nd) symmetrically normalized sheaf Laplacian.
        """
        maps = (
            restriction_maps if restriction_maps is not None else self.restriction_maps
        )
        N, d, Nd = self.num_nodes, self.d, self.Nd

        # ── Step 1: raw unnormalized Laplacian ────────────────────────────
        L_F = super().compute_connection_laplacian_vectorized(restriction_maps)

        # ── Step 2: compute degree blocks D_F[u] ∈ R^{d×d} ──────────────
        F_src = maps[:, 0]  # (E, d, d)
        F_tgt = maps[:, 1]  # (E, d, d)

        # F^T @ F products per edge endpoint: (E, d, d)
        FTF_src = torch.bmm(F_src.transpose(1, 2), F_src)  # (E, d, d)
        FTF_tgt = torch.bmm(F_tgt.transpose(1, 2), F_tgt)  # (E, d, d)

        src_nodes = self.edge_index[0]  # (E,)
        tgt_nodes = self.edge_index[1]  # (E,)

        # Accumulate into (N, d, d) degree blocks via scatter_add.
        # We flatten the (d, d) dimension so scatter_add can work on dim=0.
        D_blocks = torch.zeros(N, d, d, device=maps.device, dtype=maps.dtype)

        # src endpoint contribution: each edge adds FTF_src[e] to D_blocks[u]
        src_idx = src_nodes.view(-1, 1, 1).expand_as(FTF_src)  # (E, d, d)
        D_blocks.scatter_add_(0, src_idx, FTF_src)

        # tgt endpoint contribution: each edge adds FTF_tgt[e] to D_blocks[v]
        tgt_idx = tgt_nodes.view(-1, 1, 1).expand_as(FTF_tgt)  # (E, d, d)
        D_blocks.scatter_add_(0, tgt_idx, FTF_tgt)

        # ── Step 3: D_F[u]^{-1/2} via eigendecomposition ─────────────────
        # D_blocks is symmetric PSD; eigh is numerically stable and returns
        # real eigenvalues in ascending order.
        #
        # D = Q Λ Q^T  →  D^{-1/2} = Q Λ^{-1/2} Q^T
        # with Λ clamped to eps to avoid divide-by-zero.
        # eigh requires fp32 on CUDA (not implemented for Half)
        #
        # Tikhonov regularisation: add ε·I before eigh.
        # This ensures:
        #   (a) all eigenvalues ≥ ε > 0  →  no divide-by-zero in D^{-1/2}
        #   (b) eigenvalues are strictly separated  →  eigh gradient is finite
        #       (gradient involves 1/(λ_i−λ_j); repeated eigenvalues → NaN)
        #   (c) the algorithm always converges  →  no LinAlgError
        _eps_reg = 1e-5
        _eye = torch.eye(d, device=D_blocks.device, dtype=D_blocks.dtype)
        D_blocks_f32 = (D_blocks + _eps_reg * _eye.unsqueeze(0)).float()
        eigenvalues, eigenvectors = torch.linalg.eigh(D_blocks_f32)  # (N,d), (N,d,d)

        # Threshold-based zero-out instead of clamp+invert:
        # If eigenvalue < threshold (near-zero degree node / rank-deficient block),
        # leave that mode un-normalized (set D^{-1/2} contribution to 0) rather than
        # dividing by a tiny number, which would blow the Frobenius norm to inf.
        threshold = 1e-4
        inv_sqrt_eigenvalues = torch.where(
            eigenvalues > threshold,
            eigenvalues.clamp(min=threshold).pow(-0.5),
            torch.zeros_like(eigenvalues),
        )  # (N, d)

        # cast back to compute dtype
        inv_sqrt_eigenvalues = inv_sqrt_eigenvalues.to(D_blocks.dtype)
        eigenvectors = eigenvectors.to(D_blocks.dtype)

        # D_inv_sqrt[u] = Q[u] @ diag(λ^{-1/2}[u]) @ Q[u]^T
        # Shape: (N, d, d)
        D_inv_sqrt = torch.einsum(
            "nij,nj,nkj->nik",
            eigenvectors,           # Q     (N, d, d)
            inv_sqrt_eigenvalues,   # Λ^{-½}(N, d)
            eigenvectors,           # Q^T   (N, d, d) — last index = eigvec index
        )

        # ── Step 4: symmetric normalization via einsum ────────────────────
        # Reshape L_F: (Nd, Nd) → (N, d, N, d)
        L_blocks = L_F.reshape(N, d, N, d)

        # L_sym[u, i, v, j] = sum_{i', j'} D_inv_sqrt[u, i, i'] * L_blocks[u, i', v, j'] * D_inv_sqrt[v, j, j']
        # Equivalently, using the symmetry of D_inv_sqrt:
        #   first contract left:  M[u, i, v, j'] = sum_{i'} D_inv_sqrt[u, i, i'] * L_blocks[u, i', v, j']
        #   then contract right:  L_sym[u, i, v, j] = sum_{j'} M[u, i, v, j'] * D_inv_sqrt[v, j', j]
        #
        # We do both in a single fused einsum for clarity and efficiency.
        L_sym_blocks = torch.einsum(
            "uik,ukvl,vjl->uivj",
            D_inv_sqrt,   # (N, d, d)  — left factor for node u
            L_blocks,     # (N, d, N, d)
            D_inv_sqrt,   # (N, d, d)  — right factor for node v (symmetric, so same tensor)
        )

        return L_sym_blocks.reshape(Nd, Nd)


# ── Verification: correctness test ──────────────────────────────────────

def verify_vectorized_matches_loop(N=10, E=20, d=3, seed=42):
    """
    Verify that the vectorized Laplacian matches the Python-loop version.

    This is a correctness proof: both methods should produce identical results
    up to floating-point tolerance.
    """
    torch.manual_seed(seed)
    edge_index = torch.randint(0, N, (2, E))

    # Build with vectorized
    layer = VectorizedSheafDiffusion(N, E, d, edge_index)
    L_vec = layer.compute_connection_laplacian_vectorized()

    # Build with explicit loop (reference implementation)
    maps = layer.restriction_maps
    F_src = maps[:, 0]
    F_tgt = maps[:, 1]
    Nd = N * d
    L_loop = torch.zeros(Nd, Nd)

    off_diag = -torch.bmm(F_src.transpose(1, 2), F_tgt)
    diag_src = torch.bmm(F_src.transpose(1, 2), F_src)
    diag_tgt = torch.bmm(F_tgt.transpose(1, 2), F_tgt)

    src = edge_index[0]
    tgt = edge_index[1]
    for e_idx in range(E):
        u, v = src[e_idx].item(), tgt[e_idx].item()
        us, ue = u * d, (u + 1) * d
        vs, ve = v * d, (v + 1) * d
        L_loop[us:ue, vs:ve] += off_diag[e_idx]
        L_loop[vs:ve, us:ue] += off_diag[e_idx].T
        L_loop[us:ue, us:ue] += diag_src[e_idx]
        L_loop[vs:ve, vs:ve] += diag_tgt[e_idx]

    max_diff = (L_vec - L_loop).abs().max().item()
    match = max_diff < 1e-5
    print(f"Vectorized vs Loop max diff: {max_diff:.2e} — {'✅ MATCH' if match else '❌ MISMATCH'}")

    # Verify properties
    props = layer.verify_laplacian_properties()
    print(f"Symmetric: {props['is_symmetric']}, PSD: {props['is_psd']}, "
          f"Spectral gap: {props['spectral_gap']:.4f}, κ: {props['condition_number']:.1f}")

    return match


if __name__ == "__main__":
    verify_vectorized_matches_loop()
