"""GNN-augmented variational stochastic block model (GNN-vSBM).

Extends the low-rank vSBM in ``hidden_markov_graph.py``. Soft assignments
``alpha = softmax(beta)`` feed a low-rank bilinear edge model, and a directed
GNN supplies a *residual* multi-hop correction:

    eta_ij = (alpha_i U_s)·(alpha_j U_t) + b
             + gamma * (h_i^{src} · h_j^{tgt})

Hop features are combined with Jumping Knowledge (concat of
``h^{(0)},…,h^{(L)}`` then a linear map) so deeper layers expand the
receptive field without erasing 0-hop cluster identity.

The only free per-node parameter is ``q_logits``: unlike the ``+e`` variants
there is no residual *table* that can explain an edge while bypassing cluster
identity. That is a weaker guarantee than it looks. Measured at $d=64$, $k=729$,
lr $0.1$ under the constraints described below, held-out AUC rises monotonically
with depth ($0.967 \to 0.974 \to 0.980 \to 0.980$ for $L = 0, 1, 2, 4$) while
agreement with the ground-truth types falls by roughly half over the same range
(Hungarian $11440 \to 6177 \to 6028 \to 4548$, NMI $0.53 \to 0.40 \to 0.42 \to
0.38$), and the decoder bias is driven from $-6.7$ to $-20$ to make room for the
residual. Multi-hop mixing of $\alpha U$ evidently recovers enough
neighbourhood-specific signal to predict edges without the assignments having to
carry type identity. Selection on held-out likelihood therefore prefers the
deepest arm and the worst clustering; the ``gt_*`` diagnostics are the only
warning of it.

Minibatches are square induced subgraphs drawn by
[`subgraph_sampler.SubgraphBatchSampler`](subgraph_sampler.py), matching the LV
trainers. With ``--likelihood {poisson,nb}`` the adjacency is *not* binarized
and ``eta_ij`` is read as a log-rate; message passing always uses the binary
pattern. Model selection remains the held-out **Bernoulli** log-likelihood so
that ``val_metric`` is comparable across every variant.

Training vs. evaluation propagation
-----------------------------------
Training is *subgraph-local* (GraphSAINT / Cluster-GCN style): every layer
propagates over the induced subgraph on the sampled node set $S$ only, and
``q_logits`` is indexed down to $S$ **before** the softmax, so the per-step cost
is $O(|S|)$ rather than $O(n)$. Evaluation, held-out scoring and assignment
extraction always run one exact full-graph propagation over all $n$ nodes. The
two paths are separate methods (``encode_subgraph`` / ``encode_full``) and the
full-graph path asserts that it produced embeddings for every node, so the
distinction cannot silently drift.

Boundary nodes of $S$ see truncated neighborhoods, which is the accepted cost of
this scheme; no GraphSAINT normalization coefficients or halo hops are applied.

As in the LV trainers, the reported model is the epoch attaining the best
``val_metric`` rather than the last one, and ground-truth agreement is recorded
at both that checkpoint (``gt_*``) and the final epoch (``last_gt_*``) as a
diagnostic that never feeds selection. ``--label-smoothing`` is offered here for
parity with those trainers: it caps the optimal Bernoulli logit at a finite
value, applies to the **training loss only**, and is inert under the count
likelihoods.

What ``--u-norm unit`` does and does not bound
----------------------------------------------
The block logit splits as ``eta = eta_LV + eta_GNN`` (see ``block_terms``), and
the constraint reaches exactly one of the two halves:

``eta_LV = (alpha_i U_s)·(alpha_j U_t) + b``
    Bounded. ``u_src`` and ``u_tgt`` are unit-normalised row-wise in the forward
    pass and the left factor is multiplied by the scale ``s``. Since ``alpha_i``
    lies in the simplex, ``||alpha_i U_s|| <= s`` and ``||alpha_j U_t|| <= 1``, so
    ``|eta_LV - b| <= s`` holds structurally, at every one of the three places
    the decoder is evaluated: the subgraph-local block, the full-propagation
    block, and the held-out pair path.
``eta_GNN = gamma * (h_i · h_j)``
    Not bounded by ``--u-norm``. Neither ``gamma``, the JK projection, nor the
    source/target heads are constrained, so this term can grow without limit even
    under ``--u-norm unit`` — and measurably does: at $L=2$, $d=64$,
    $\\mathrm{lr}=0.01$, $\\max|\\eta^{GNN}|$ climbs from $7$ to $106$ within four
    epochs while $\\max|\\eta^{LV}|$ never leaves $[6, 11]$, and it is the residual
    that carries the run away. ``--gnn-norm unit`` normalises the two head outputs
    to unit rows in the forward pass, which turns the residual into a cosine
    similarity and makes ``gamma`` its entire magnitude budget:
    $|\\eta^{GNN}| \\le |\\gamma|$.

The per-epoch log reports ``max_abs_lv`` and ``max_abs_residual`` alongside
``max_abs_logit``, so which half is growing is never a matter of inference.

Note that a small ``gamma`` is *not* evidence that the residual is inactive. The
term is a product, and with the heads unconstrained the optimiser is free to
satisfy the data by growing $\\lVert h\\rVert$ instead of ``gamma``: the run above
reaches $|\\eta^{GNN}| = 106$ at $\\gamma = -0.003$. Only under ``--gnn-norm unit``
does ``gamma`` measure what the GNN actually contributes.

``--u-norm none --gnn-norm none`` (the defaults) is the historical decoder, bit
for bit.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, load_npz
from torch import nn
from tqdm import tqdm

from evaluate_clustering import score_assignment_dict
from heldout import (
    LIKELIHOODS,
    heldout_pair_mask,
    load_heldout,
    lookup_pair_counts,
    mean_auc,
    mean_bernoulli_ll,
    native_ll_name,
    nb_edge_logit,
    nb_log_pmf,
    poisson_edge_logit,
    poisson_log_pmf,
    remove_heldout_positives,
    write_metrics,
)
from index_mapping import load_mapping
from subgraph_sampler import SubgraphBatchSampler
from training_utils import (
    GNN_NORMS,
    LABEL_SMOOTHING_TARGETS,
    BestCheckpoint,
    add_u_norm_arguments,
    block_scale_metrics,
    checkpoint_metrics,
    decoder_line,
    label_smoothing_line,
    make_block_scale,
    prefixed,
    resolve_grad_clip,
    resolve_label_smoothing,
    safe_optimizer_step,
    scaled_block_embeddings,
    smooth_binary_targets,
    u_norm_line,
    unit_rows,
)


class Snapshot(NamedTuple):
    """Everything reported about one parameter state (best checkpoint or last epoch)."""

    assignments: np.ndarray
    scores: np.ndarray
    u: np.ndarray
    assignment_dict: dict
    val_ll: float
    native_ll: float
    val_auc: float
    gamma: float
    gt: dict[str, float] | None


# Empirically measured undirected hop-ball sizes (excl. self) on FlyWire A.
HOP_BALL_NOTE = (
    "Receptive field (undirected hop ball, median excl. self): "
    "L=1 ~18, L=2 ~3k, L=3 ~37k, L=4 ~105k nodes."
)
PROPAGATIONS = ("subgraph", "full")


def gnn_norm_line(gnn_norm: str, gamma_init: float) -> str:
    """One-line description of the residual constraint, for the run header."""
    if gnn_norm == "none":
        return "gnn_norm=none (residual gamma*(h_i.h_j) unbounded; |gamma| does not bound it)"
    return (
        f"gnn_norm=unit (residual is a cosine similarity) => |eta_GNN| <= |gamma|, "
        f"{abs(gamma_init):.4f} at init"
    )


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def log_clamp(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def csr_to_torch_sparse(mat: csr_matrix, device: str, dtype: torch.dtype) -> torch.Tensor:
    coo = mat.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=dtype, device=device)
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device).coalesce()


def row_normalize_sparse(adj: torch.Tensor) -> torch.Tensor:
    """Row-normalize a coalesced sparse adjacency (out-degree normalization)."""
    indices = adj.indices()
    values = adj.values()
    row = indices[0]
    deg = torch.zeros(adj.size(0), dtype=values.dtype, device=values.device)
    deg.index_add_(0, row, values)
    inv_deg = 1.0 / deg.clamp(min=1.0)
    norm_values = values * inv_deg[row]
    return torch.sparse_coo_tensor(
        indices, norm_values, size=adj.size(), device=adj.device
    ).coalesce()


def row_normalize_dense(block: torch.Tensor) -> torch.Tensor:
    """Row-normalize a dense adjacency block, matching ``row_normalize_sparse``."""
    return block / block.sum(dim=1, keepdim=True).clamp(min=1.0)


def transpose_sparse(adj: torch.Tensor) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        adj.indices().flip(0),
        adj.values(),
        size=adj.size(),
        device=adj.device,
    ).coalesce()


def batch_membership(n: int, batch_idx: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    flags = torch.zeros(n, dtype=torch.bool, device=device)
    flags[batch_idx] = True
    return flags


def internal_edge_keep(adj: torch.Tensor, in_batch: torch.Tensor) -> torch.Tensor:
    """Boolean keep-mask dropping directed edges with both endpoints in the batch."""
    indices = adj.indices()
    return ~(in_batch[indices[0]] & in_batch[indices[1]])


def apply_edge_keep(adj: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    if bool(keep.all()):
        return adj
    indices = adj.indices()
    return torch.sparse_coo_tensor(
        indices[:, keep],
        adj.values()[keep],
        size=adj.size(),
        device=adj.device,
    ).coalesce()


def mask_minibatch_edges(adj: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """Zero directed edges whose both endpoints lie in ``batch_idx``.

    Only used by ``--propagation full``: it keeps the full-graph encoder from
    reading the very edges the decoder must reconstruct. It is meaningless under
    subgraph-local propagation, where those edges *are* the propagation graph.
    """
    in_batch = batch_membership(adj.size(0), batch_idx, adj.device)
    return apply_edge_keep(adj, internal_edge_keep(adj, in_batch))


def spmm(a: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Propagate ``h`` through ``a``; ``a`` is sparse (full graph) or dense (block)."""
    return torch.sparse.mm(a, h) if a.is_sparse else a @ h


def pair_log_lik(
    likelihood: str, eta: torch.Tensor, target: torch.Tensor, log_r: torch.Tensor
) -> torch.Tensor:
    """Per-pair log-likelihood; ``eta`` is a logit (Bernoulli) or a log-rate."""
    if likelihood == "bernoulli":
        return -F.binary_cross_entropy_with_logits(eta, target, reduction="none")
    if likelihood == "poisson":
        return poisson_log_pmf(target, eta)
    return nb_log_pmf(target, eta, log_r)


def pair_edge_logit(likelihood: str, eta: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
    """Logit of $P(y>0)$ implied by the fitted likelihood."""
    if likelihood == "bernoulli":
        return eta
    if likelihood == "poisson":
        return poisson_edge_logit(eta)
    return nb_edge_logit(eta, log_r)


class DirectedGNNLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w_self = nn.Linear(d, d, bias=False)
        self.w_in = nn.Linear(d, d, bias=False)
        self.w_out = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.act = nn.GELU()

    def forward(
        self,
        h: torch.Tensor,
        a_out: torch.Tensor,
        a_in: torch.Tensor,
    ) -> torch.Tensor:
        msg_out = spmm(a_out, h)
        msg_in = spmm(a_in, h)
        update = self.w_self(h) + self.w_out(msg_out) + self.w_in(msg_in)
        return self.act(self.norm(h + update))


class GNNvSBM(nn.Module):
    """Mean-field vSBM with residual multi-hop GNN correction + JK."""

    def __init__(
        self,
        n: int,
        k: int,
        d: int,
        n_layers: int,
        edge_bias_init: float,
        dtype: torch.dtype,
        gamma_init: float = 0.0,
        u_norm: str = "none",
        u_scale: str = "fixed",
        u_scale_init: float = 1.0,
        gnn_norm: str = "none",
    ):
        super().__init__()
        if gnn_norm not in GNN_NORMS:
            raise ValueError(f"unknown gnn-norm {gnn_norm!r}; expected one of {GNN_NORMS}")
        self.gnn_norm = gnn_norm
        self.n = n
        self.k = k
        self.d = d
        self.n_layers = n_layers
        self.q_logits = nn.Parameter((1.0 / k) * torch.randn(n, k, dtype=dtype))
        # Shared projection for GNN node features (0-hop).
        self.u = nn.Parameter((1.0 / np.sqrt(k * d)) * torch.randn(k, d, dtype=dtype))
        # Low-rank LV prototypes (bilinear decoder backbone).
        self.u_src = nn.Parameter((1.0 / np.sqrt(k * d)) * torch.randn(k, d, dtype=dtype))
        self.u_tgt = nn.Parameter((1.0 / np.sqrt(k * d)) * torch.randn(k, d, dtype=dtype))
        self.layers = nn.ModuleList([DirectedGNNLayer(d) for _ in range(n_layers)])
        # Jumping Knowledge: concat h0..hL then project back to d.
        self.jk_proj = nn.Linear(d * (n_layers + 1), d, bias=False)
        self.src_head = nn.Linear(d, d, bias=False)
        self.tgt_head = nn.Linear(d, d, bias=False)
        self.bias = nn.Parameter(torch.tensor([edge_bias_init], dtype=dtype))
        # Residual mix; init near 0 so training can fall back to pure LV.
        self.gamma = nn.Parameter(torch.tensor([gamma_init], dtype=dtype))
        # Negative-binomial dispersion; only optimized under --likelihood nb.
        self.log_r = nn.Parameter(torch.zeros(1, dtype=dtype))
        # Parameters are tracked only when assigned directly as attributes, so the
        # scale is registered here even though it is held by a plain helper object.
        self.block_scale = make_block_scale(u_norm, u_scale, u_scale_init, k, dtype, "cpu")
        if self.block_scale is not None and self.block_scale.log_scale is not None:
            self.u_log_scale = self.block_scale.log_scale

    def _apply(self, *args, **kwargs):
        """Keep the ``BlockScale`` pointing at the parameter the optimizer holds.

        ``nn.Module.to`` only mutates a parameter in place when the conversion is
        shallow-copy compatible; any change of device installs a *new* ``Parameter``
        in ``_parameters`` instead. ``BlockScale`` keeps its own reference, so
        without this rebind the forward pass would read a stale CPU tensor while
        ``named_parameters`` (hence the optimizer and the checkpoint) tracked the
        device copy, and the learned scale would never move. Silent on CPU,
        fatal on an accelerator, which is exactly the wrong way round.
        """
        module = super()._apply(*args, **kwargs)
        if module.block_scale is not None and module.block_scale.log_scale is not None:
            module.block_scale.log_scale = module.u_log_scale
        return module

    def prototypes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Low-rank prototypes as they enter the decoder, normalised if requested."""
        return scaled_block_embeddings(self.u_src, self.u_tgt, self.block_scale)

    def soft_assignments(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.q_logits if idx is None else self.q_logits[idx]
        return torch.softmax(logits, dim=-1)

    def propagate(
        self,
        h: torch.Tensor,
        a_out: torch.Tensor,
        a_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the JK stack over whichever graph ``a_out``/``a_in`` describe."""
        hops = [h]
        for layer in self.layers:
            h = layer(h, a_out, a_in)
            hops.append(h)
        h_jk = self.jk_proj(torch.cat(hops, dim=-1))
        return self.src_head(h_jk), self.tgt_head(h_jk)

    def encode_full(
        self,
        a_out: torch.Tensor,
        a_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact all-node embeddings. Evaluation / assignment extraction only."""
        h_src, h_tgt = self.propagate(self.soft_assignments() @ self.u, a_out, a_in)
        assert h_src.shape[0] == self.n, "encode_full must cover every node"
        return h_src, h_tgt

    def encode_subgraph(
        self,
        idx: torch.Tensor,
        a_out_sub: torch.Tensor,
        a_in_sub: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training-only embeddings on the induced subgraph over ``idx``.

        ``q_logits`` is indexed down to the block *before* the softmax and the
        projection, so no $n\\times K$ tensor is ever materialized here.
        """
        alpha = self.soft_assignments(idx)
        h_src, h_tgt = self.propagate(alpha @ self.u, a_out_sub, a_in_sub)
        return alpha, h_src, h_tgt

    def block_terms(
        self,
        alpha_i: torch.Tensor,
        alpha_j: torch.Tensor,
        g_src: torch.Tensor,
        g_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two halves of the dense block logit, $\\eta = \\eta^{LV} + \\eta^{GNN}$.

        Returned separately because only the first is constrained: under
        ``--u-norm unit`` the prototypes are unit rows times $s$, so
        $|\\eta^{LV} - b| \\le s$, whereas $\\eta^{GNN} = \\gamma\\,h_i^\\top h_j$
        involves no normalised tensor and is bounded by nothing. Which of the two
        is growing is therefore the diagnostic that decides whether the constraint
        can be expected to help at all.

        Callers pass block-local embeddings under ``--propagation subgraph`` and
        rows of the full-graph embeddings under ``--propagation full``; the
        arithmetic is identical either way.
        """
        u_src, u_tgt = self.prototypes()
        lv = (alpha_i @ u_src) @ (alpha_j @ u_tgt).T
        g_src, g_tgt = self.residual_embeddings(g_src, g_tgt)
        return lv + self.bias, self.gamma * (g_src @ g_tgt.T)

    def residual_embeddings(
        self, g_src: torch.Tensor, g_tgt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Head outputs as they enter the residual, normalised under ``--gnn-norm unit``.

        Normalising here rather than at the heads keeps ``--gnn-norm none`` an
        exact identity, and puts the constraint on the tensors that actually meet
        in the inner product: with unit rows $|h_i^\\top h_j| \\le 1$, so
        $|\\eta^{GNN}| \\le |\\gamma|$ at every call site.
        """
        if self.gnn_norm == "none":
            return g_src, g_tgt
        return unit_rows(g_src), unit_rows(g_tgt)

    def pair_logits(
        self,
        h_src: torch.Tensor,
        h_tgt: torch.Tensor,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Per-pair logits for held-out evaluation (not a dense block)."""
        alpha_s = self.soft_assignments(src)
        alpha_t = self.soft_assignments(tgt)
        u_src, u_tgt = self.prototypes()
        lv = ((alpha_s @ u_src) * (alpha_t @ u_tgt)).sum(dim=-1)
        g_src, g_tgt = self.residual_embeddings(h_src[src], h_tgt[tgt])
        gnn = (g_src * g_tgt).sum(dim=-1)
        return lv + self.bias + self.gamma * gnn


def load_adjacency(path: str, likelihood: str) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Return ``(counts, decoder target, binary pattern)``.

    The decoder is trained on raw synapse counts under Poisson / NB and on the
    binarized pattern under Bernoulli; message passing always uses the binary
    pattern so that propagation is unaffected by the observation model.
    """
    counts = load_npz(path).tocsr().astype(np.float32)
    counts.eliminate_zeros()
    binary = counts.copy()
    binary.data = (binary.data > 0).astype(np.float32)
    binary.eliminate_zeros()
    return counts, (counts if likelihood != "bernoulli" else binary), binary


def train(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    dtype = torch.float32
    torch.manual_seed(args.seed)

    adj_counts, adj, adj_bin = load_adjacency(args.adjacency, args.likelihood)
    n = adj_bin.shape[0]
    nnz = int(adj_bin.nnz)
    base_rate = max(float(adj_bin.mean()), 1e-8)
    label_smoothing = resolve_label_smoothing(args.label_smoothing, args.likelihood)
    print(f"Loaded A with shape {adj_bin.shape}, nnz={nnz}, device={device}")
    print(HOP_BALL_NOTE)
    print(
        f"GNN: L={args.layers} (JK over hops 0..{args.layers}) d={args.d} lr={args.lr} "
        f"likelihood={args.likelihood} bfs_frac={args.bfs_frac} bfs_seeds={args.bfs_seeds} "
        f"gamma_init={args.gamma_init} propagation={args.propagation}"
    )
    print(label_smoothing_line(label_smoothing, base_rate, args.label_smoothing_target))
    if args.propagation == "full":
        print(f"Full-graph propagation; minibatch edge masking={not args.no_edge_mask}")

    held = load_heldout(Path(args.heldout_pairs))
    held_src_np = held["src"]
    held_tgt_np = held["tgt"]
    held_counts = (
        None
        if args.likelihood == "bernoulli"
        else lookup_pair_counts(adj_counts, held_src_np, held_tgt_np)
    )

    # Message passing graph: remove held-out positive edges to avoid leakage.
    # Evaluation always propagates over this graph, in full.
    adj_mp = remove_heldout_positives(adj_bin, held_src_np, held_tgt_np, held["y"])
    a_bin = csr_to_torch_sparse(adj_mp, device=device, dtype=dtype)
    a_bin_t = transpose_sparse(a_bin)
    a_out_full = row_normalize_sparse(a_bin)
    a_in_full = row_normalize_sparse(a_bin_t)
    mp_nnz = int(a_bin._nnz())
    full_mean_out_degree = mp_nnz / max(n, 1)

    edge_bias_init = float(np.log(base_rate))
    model = GNNvSBM(
        n=n,
        k=args.k,
        d=args.d,
        n_layers=args.layers,
        edge_bias_init=edge_bias_init,
        dtype=dtype,
        gamma_init=args.gamma_init,
        u_norm=args.u_norm,
        u_scale=args.u_scale,
        u_scale_init=args.u_scale_init,
        gnn_norm=args.gnn_norm,
    ).to(device=device, dtype=dtype)
    print(u_norm_line(args.u_norm, model.block_scale, edge_bias_init))
    print(gnn_norm_line(args.gnn_norm, args.gamma_init))

    params = [p for name, p in model.named_parameters() if name != "log_r"]
    if args.likelihood == "nb":
        params.append(model.log_r)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    grad_clip = resolve_grad_clip(args.grad_clip)
    state = dict(model.named_parameters())
    sampler = SubgraphBatchSampler(
        adj_bin,
        batch_size=args.minibatch,
        bfs_frac=args.bfs_frac,
        n_seeds=args.bfs_seeds,
        seed=args.seed,
    )
    checkpoint = BestCheckpoint()
    cum_pos_obs = 0
    total_updates = 0
    n_skipped = 0
    last_epoch = -1
    max_abs_eta = 0.0
    max_abs_lv = 0.0
    max_abs_res = 0.0
    # A coverage-defined epoch is longer at higher bfs_frac, so an update budget is
    # what makes compute comparable across bfs_frac.
    epochs = args.epochs if args.target_updates is None else 10**9

    try:
        for epoch in range(epochs):
            if args.target_updates is not None and total_updates >= args.target_updates:
                break
            loss_sum = 0.0
            n_scored = 0
            n_batches = 0
            epoch_pos_obs = 0
            epoch_max_abs_eta = 0.0
            epoch_max_abs_lv = 0.0
            epoch_max_abs_res = 0.0
            block_edges = 0
            block_nodes = 0

            for idx in tqdm(sampler.epoch(), desc=f"gnn epoch {epoch}"):
                if args.max_updates is not None and n_batches >= args.max_updates:
                    break
                if args.target_updates is not None and total_updates >= args.target_updates:
                    break
                block = torch.tensor(adj[idx][:, idx].toarray(), dtype=dtype, device=device)
                mask = torch.tensor(
                    heldout_pair_mask(idx, idx, held_src_np, held_tgt_np), device=device
                )
                keep = ~mask
                # The graph has no self-loops; do not train on the diagonal.
                keep.fill_diagonal_(False)
                keep = keep.to(dtype)
                n_keep = keep.sum()
                if float(n_keep.item()) == 0.0:
                    continue
                n_batches += 1
                total_updates += 1
                # Binary in-block pattern with held-out positives removed: this is
                # exactly the induced subgraph of the message-passing graph.
                prop_block = (block > 0).to(dtype) * keep
                n_block_edges = int(prop_block.sum().item())
                epoch_pos_obs += n_block_edges
                block_edges += n_block_edges
                block_nodes += int(idx.size)

                idx_t = torch.from_numpy(idx).to(device)
                if args.propagation == "subgraph":
                    alpha, h_src, h_tgt = model.encode_subgraph(
                        idx_t,
                        row_normalize_dense(prop_block),
                        row_normalize_dense(prop_block.T),
                    )
                    lv_eta, res_eta = model.block_terms(alpha, alpha, h_src, h_tgt)
                else:
                    if args.no_edge_mask or args.layers == 0:
                        a_out, a_in = a_out_full, a_in_full
                    else:
                        in_batch = batch_membership(n, idx_t, device)
                        a_out = row_normalize_sparse(
                            apply_edge_keep(a_bin, internal_edge_keep(a_bin, in_batch))
                        )
                        a_in = row_normalize_sparse(
                            apply_edge_keep(a_bin_t, internal_edge_keep(a_bin_t, in_batch))
                        )
                    h_src, h_tgt = model.encode_full(a_out, a_in)
                    alpha = model.soft_assignments(idx_t)
                    # The two sides are softmaxed independently, as the historical
                    # full-propagation decoder did. Sharing a single node here is
                    # algebraically identical but accumulates the gradient into
                    # q_logits in a different order, which moves the run in its
                    # last significant digits and so is not a no-op.
                    lv_eta, res_eta = model.block_terms(
                        model.soft_assignments(idx_t),
                        model.soft_assignments(idx_t),
                        h_src[idx_t],
                        h_tgt[idx_t],
                    )

                eta = lv_eta + res_eta
                epoch_max_abs_eta = max(epoch_max_abs_eta, float(eta.detach().abs().max().item()))
                epoch_max_abs_lv = max(epoch_max_abs_lv, float(lv_eta.detach().abs().max().item()))
                epoch_max_abs_res = max(
                    epoch_max_abs_res, float(res_eta.detach().abs().max().item())
                )
                # Training targets only; the held-out labels are never smoothed.
                target = smooth_binary_targets(
                    block, label_smoothing, base_rate, args.label_smoothing_target
                )
                ll_pairs = pair_log_lik(args.likelihood, eta, target, model.log_r)
                ll = ((ll_pairs * keep).sum() / n_keep.clamp(min=1.0)) * float(idx.size * idx.size)
                entropy = args.entropy_weight * -(alpha * log_clamp(alpha)).sum(1).mean()
                loss = -(ll + entropy)
                # Without this guard the first bad gradient poisons every parameter
                # and the run silently reports NaN metrics instead of failing loudly.
                if not safe_optimizer_step(loss, params, optimizer, grad_clip):
                    n_skipped += 1
                    continue
                loss_sum += float(loss.item())
                n_scored += 1

            mean_loss = loss_sum / max(n_scored, 1)
            cum_pos_obs += epoch_pos_obs
            max_abs_eta = max(max_abs_eta, epoch_max_abs_eta)
            max_abs_lv = max(max_abs_lv, epoch_max_abs_lv)
            max_abs_res = max(max_abs_res, epoch_max_abs_res)
            last_epoch = epoch
            print(sampler.coverage_line(epoch, n_batches, epoch_pos_obs, cum_pos_obs, nnz))
            block_deg = block_edges / max(block_nodes, 1)
            print(
                f"[subgraph] epoch={epoch} edges_per_block={block_edges / max(n_batches, 1):.1f} "
                f"block_out_degree={block_deg:.2f} full_out_degree={full_mean_out_degree:.2f} "
                f"degree_ratio={block_deg / max(full_mean_out_degree, 1e-9):.3f} "
                f"block_edge_frac_of_nnz="
                f"{block_edges / max(n_batches, 1) / max(mp_nnz, 1):.6f}"
            )
            val_ll, _, val_auc = eval_heldout(
                model, a_out_full, a_in_full, held, None, args.likelihood, device, dtype
            )
            checkpoint.update(val_ll, epoch, state)
            with torch.no_grad():
                n_used = int(torch.unique(torch.argmax(model.q_logits, dim=-1)).numel())
                gamma = float(model.gamma.item())
            print(f"[gamma] epoch={epoch} gamma={gamma:.6f}")
            print(decoder_line(epoch, float(model.bias.item()), model.block_scale))
            print(
                f"loss={mean_loss:.4f} val_ll={val_ll:.6f} val_auc={val_auc:.4f} "
                f"best_val_ll={checkpoint.metric:.6f}@{checkpoint.epoch} gamma={gamma:.6f} "
                f"clusters={n_used}/{args.k} max_abs_logit={epoch_max_abs_eta:.3f} "
                f"max_abs_lv={epoch_max_abs_lv:.3f} max_abs_residual={epoch_max_abs_res:.3f} "
                f"updates={total_updates} skipped={n_skipped}"
            )
    except KeyboardInterrupt:
        print("Training interrupted.")

    return save_results(
        model,
        Path(args.mapping),
        Path(args.out_prefix),
        a_out_full,
        a_in_full,
        held,
        held_counts,
        device,
        dtype,
        cum_pos_obs,
        nnz,
        total_updates,
        n_skipped,
        grad_clip,
        label_smoothing,
        base_rate,
        max_abs_eta,
        max_abs_lv,
        max_abs_res,
        checkpoint,
        state,
        last_epoch,
        args,
    )


@torch.no_grad()
def eval_heldout(
    model: GNNvSBM,
    a_out: torch.Tensor,
    a_in: torch.Tensor,
    held: dict[str, np.ndarray],
    counts: np.ndarray | None,
    likelihood: str,
    device: str,
    dtype: torch.dtype,
    chunk: int = 8192,
) -> tuple[float, float, float]:
    """Held-out (Bernoulli LL, native LL, AUC) under one exact full-graph pass.

    INVARIANT: this function scores the *true* unsmoothed $0/1$ labels
    ``held["y"]``. Label smoothing is a property of the training loss alone and
    must never reach this path -- note that it takes no smoothing argument, and
    do not add one. ``val_metric`` and ``val_auc`` are therefore directly
    comparable across every run ever made, smoothed or not.
    """
    h_src, h_tgt = model.encode_full(a_out, a_in)
    src = torch.tensor(held["src"], dtype=torch.long, device=device)
    tgt = torch.tensor(held["tgt"], dtype=torch.long, device=device)
    y = torch.tensor(held["y"], dtype=dtype, device=device)
    y_count = None if counts is None else torch.tensor(counts, dtype=dtype, device=device)
    lls: list[float] = []
    native: list[float] = []
    edge_logits: list[torch.Tensor] = []
    for start in range(0, src.numel(), chunk):
        sl = slice(start, start + chunk)
        eta = model.pair_logits(h_src, h_tgt, src[sl], tgt[sl])
        edge_logit = pair_edge_logit(likelihood, eta, model.log_r)
        lls.append(mean_bernoulli_ll(edge_logit, y[sl]))
        edge_logits.append(edge_logit)
        if y_count is not None:
            native.append(
                float(pair_log_lik(likelihood, eta, y_count[sl], model.log_r).mean().item())
            )
    bern_ll = float(np.mean(lls))
    auc = mean_auc(torch.cat(edge_logits), y)
    return bern_ll, float(np.mean(native)) if native else bern_ll, auc


def save_results(
    model: GNNvSBM,
    mapping_path: Path,
    out_prefix: Path,
    a_out: torch.Tensor,
    a_in: torch.Tensor,
    held: dict[str, np.ndarray],
    held_counts: np.ndarray | None,
    device: str,
    dtype: torch.dtype,
    cum_pos_obs: int,
    nnz: int,
    total_updates: int,
    n_skipped: int,
    grad_clip: float | None,
    label_smoothing: float,
    base_rate: float,
    max_abs_eta: float,
    max_abs_lv: float,
    max_abs_res: float,
    checkpoint: BestCheckpoint,
    state: dict[str, torch.Tensor],
    last_epoch: int,
    args: argparse.Namespace,
) -> dict:
    mapping = load_mapping(str(mapping_path))
    gt_path = None if args.no_gt_eval else args.gt

    def snapshot() -> Snapshot:
        with torch.no_grad():
            assignments = torch.argmax(model.q_logits, dim=-1).cpu().numpy()
            scores = torch.max(torch.softmax(model.q_logits, dim=-1), dim=-1).values.cpu().numpy()
            u = model.u.detach().cpu().numpy()
            val_ll, native_ll, val_auc = eval_heldout(
                model, a_out, a_in, held, held_counts, args.likelihood, device, dtype
            )
            gamma = float(model.gamma.item())
        pred = {mapping[i]: int(assignments[i]) for i in range(len(assignments))}
        return Snapshot(
            assignments,
            scores,
            u,
            pred,
            val_ll,
            native_ll,
            val_auc,
            gamma,
            score_assignment_dict(pred, gt_path),
        )

    # Evaluate the final parameters before restoring, so the cost of any mid-run
    # divergence is measurable rather than inferred.
    last = snapshot()
    last_scale = block_scale_metrics(model.block_scale, "last_")
    last_bias = float(model.bias.item())
    restored = checkpoint.epoch != last_epoch and checkpoint.restore(state)
    best = snapshot() if restored else last

    np.save(f"{out_prefix}_assignments.npy", best.assignments)
    np.save(f"{out_prefix}_scores.npy", best.scores)
    np.save(f"{out_prefix}_U.npy", best.u)
    np.save(f"{out_prefix}_assignment_dict.npy", best.assignment_dict)
    if best is not last:
        np.save(f"{out_prefix}_last_assignment_dict.npy", last.assignment_dict)
    metrics = {
        "method": "gnn_vsbm",
        "val_metric": best.val_ll,
        "val_metric_name": "heldout_bernoulli_ll",
        "val_metric_higher_is_better": True,
        "val_native_ll": best.native_ll,
        "val_native_ll_name": native_ll_name(args.likelihood),
        "val_auc": best.val_auc,
        "last_val_metric": last.val_ll,
        "last_val_native_ll": last.native_ll,
        "last_val_auc": last.val_auc,
        **checkpoint_metrics(checkpoint, last_epoch),
        **prefixed(best.gt, "gt_"),
        **prefixed(last.gt, "last_gt_"),
        "seed": args.seed,
        "k": args.k,
        "d": args.d,
        "layers": args.layers,
        "lr": args.lr,
        "gamma": best.gamma,
        "last_gamma": last.gamma,
        "gamma_init": args.gamma_init,
        "entropy_weight": args.entropy_weight,
        "epochs": args.epochs,
        "likelihood": args.likelihood,
        "bfs_frac": args.bfs_frac,
        "bfs_seeds": args.bfs_seeds,
        "propagation": args.propagation,
        "grad_clip": grad_clip,
        "label_smoothing": args.label_smoothing,
        "label_smoothing_target": args.label_smoothing_target,
        "label_smoothing_applied": label_smoothing > 0.0,
        "u_norm": args.u_norm,
        "u_scale": args.u_scale,
        "u_scale_init": args.u_scale_init,
        "gnn_norm": args.gnn_norm,
        **block_scale_metrics(model.block_scale),
        **last_scale,
        "decoder_bias": float(model.bias.item()),
        "last_decoder_bias": last_bias,
        "train_base_rate": base_rate,
        "max_abs_train_logit": max_abs_eta,
        "max_abs_train_lv_logit": max_abs_lv,
        "max_abs_train_residual": max_abs_res,
        "edges_seen_x_nnz": cum_pos_obs / max(nnz, 1),
        "total_updates": total_updates,
        "skipped_updates": n_skipped,
        "target_updates": args.target_updates,
        "nb_r": float(torch.exp(model.log_r).item()) if args.likelihood == "nb" else None,
        "n_pred_clusters": int(len(np.unique(best.assignments))),
        "last_n_pred_clusters": int(len(np.unique(last.assignments))),
    }
    write_metrics(f"{out_prefix}_metrics.json", metrics)
    print(
        f"Saved {out_prefix}_* from the {metrics['checkpoint']} model "
        f"(epoch {metrics['best_epoch']}); val_ll={best.val_ll:.6f} "
        f"native_ll={best.native_ll:.6f} val_auc={best.val_auc:.4f} gamma={best.gamma:.6f} "
        f"| last val_ll={last.val_ll:.6f} val_auc={last.val_auc:.4f}"
    )
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adjacency", default="sparse_connectivity_matrix.npz")
    p.add_argument("--mapping", default="root_id_to_index_mapping.json")
    p.add_argument("--heldout-pairs", default="heldout_pairs.npz")
    p.add_argument("--out-prefix", default="gnn")
    p.add_argument("--k", type=int, default=729)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--layers", type=int, default=2, help="GNN depth L; 0 = LV + JK(h0) only")
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-updates", type=int, default=None, help="Cap on updates per epoch")
    p.add_argument(
        "--target-updates",
        type=int,
        default=None,
        help="Total update budget, overriding --epochs; matches compute across bfs_frac",
    )
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument(
        "--bfs-frac",
        type=float,
        default=0.5,
        help="Fraction of each minibatch grown by BFS; 0 recovers uniform sampling",
    )
    p.add_argument("--bfs-seeds", type=int, default=4, help="BFS seeds per expansion round")
    p.add_argument(
        "--likelihood",
        choices=LIKELIHOODS,
        default="bernoulli",
        help="Edge likelihood; count models use unbinarized synapse counts",
    )
    p.add_argument(
        "--propagation",
        choices=PROPAGATIONS,
        default="subgraph",
        help=(
            "Training-time message passing: 'subgraph' propagates over the induced "
            "block only (GraphSAINT style, default); 'full' propagates over all "
            "nodes every step. Evaluation is always full-graph."
        ),
    )
    p.add_argument("--gamma-init", type=float, default=0.0, help="Init for residual GNN mix")
    p.add_argument("--entropy-weight", type=float, default=1.0)
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Global gradient-norm clip applied before each step; 0 or less disables it",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help=(
            "Bernoulli target smoothing eps in [0, 1); 0 (default) trains on hard "
            "0/1 targets, whose optimal logit is unbounded. Applied to the training "
            "loss only, and ignored for the count likelihoods"
        ),
    )
    p.add_argument(
        "--label-smoothing-target",
        choices=LABEL_SMOOTHING_TARGETS,
        default="base_rate",
        help=(
            "Prior the targets are pulled towards: 'base_rate' preserves the edge "
            "marginal; 'uniform' (0.5) inflates negatives far above the true density"
        ),
    )
    add_u_norm_arguments(p)
    p.add_argument(
        "--gnn-norm",
        choices=GNN_NORMS,
        default="none",
        help=(
            "Constraint on the GNN head outputs entering the residual term. "
            "'none' (default) leaves them unconstrained, reproducing every run "
            "made so far; 'unit' normalises each row in the forward pass, which "
            "makes the residual a cosine similarity and bounds |eta_GNN| by |gamma|"
        ),
    )
    p.add_argument(
        "--gt",
        default="root_id_type_dict.pkl",
        help="Ground-truth types, scored as a diagnostic only (never used for selection)",
    )
    p.add_argument(
        "--no-gt-eval",
        action="store_true",
        help="Skip the diagnostic ground-truth scoring of the best and last models",
    )
    p.add_argument(
        "--no-edge-mask",
        action="store_true",
        help="--propagation full only: keep in-block edges in the propagation graph",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.seed is None:
        args.seed = int(datetime.now().timestamp())
    train(args)


if __name__ == "__main__":
    main()
