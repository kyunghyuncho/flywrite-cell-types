"""Partner-histogram consistency for the LV SBM.

Empirical out-neighbour type mix of node $i$ is the stop-gradient average of
$q_j$ over $j$ with $A_{ij}=1$. The SBM prediction is the block row mixed by
$q_i$: $\\tilde h_i = q_i\\,\\sigma(\\eta)$. The extra term is
$\\mathrm{KL}(h_i^{\\mathrm{out}}\\Vert\\tilde h_i)$ (and the in-neighbour
analogue). This is *not* homophily: neighbours should look like the partners of
$i$'s type, not like $i$ itself.

Held-out positive edges are stripped from the adjacency used here so the
regulariser cannot see the selection pairs.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix


def log_clamp(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def binary_adjacency_without_heldout(adj: csr_matrix, held: dict[str, np.ndarray]) -> csr_matrix:
    """Binary connectivity with held-out *positive* directed pairs removed."""
    out = adj.tocoo()
    data = (out.data > 0).astype(np.float32)
    y = np.asarray(held["y"])
    pos = y > 0.5
    blocked = set(zip(np.asarray(held["src"])[pos].tolist(), np.asarray(held["tgt"])[pos].tolist()))
    keep = np.ones(out.nnz, dtype=bool)
    if blocked:
        keep = np.fromiter(
            ((int(r), int(c)) not in blocked for r, c in zip(out.row.tolist(), out.col.tolist())),
            dtype=bool,
            count=out.nnz,
        )
    return csr_matrix(
        (data[keep], (out.row[keep], out.col[keep])),
        shape=adj.shape,
        dtype=np.float32,
    )


def neighbor_type_mass(
    adj_rows: csr_matrix,
    q_logits: torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop-grad type mass of neighbours for each row of ``adj_rows`` (B × n).

    Returns ``(mass, degree)`` with ``mass[b] = sum_{j: A_bj=1} sg[q_j]``.
    """
    b, k = adj_rows.shape[0], q_logits.shape[1]
    mass = torch.zeros(b, k, device=device, dtype=dtype)
    deg = torch.zeros(b, device=device, dtype=dtype)
    if adj_rows.nnz == 0:
        return mass, deg
    coo = adj_rows.tocoo()
    rows = torch.from_numpy(coo.row.astype(np.int64)).to(device)
    cols = torch.from_numpy(coo.col.astype(np.int64)).to(device)
    w = torch.from_numpy(np.asarray(coo.data, dtype=np.float32)).to(device=device, dtype=dtype)
    q_cols = torch.softmax(q_logits[cols], dim=-1).detach()
    mass.index_add_(0, rows, q_cols * w[:, None])
    deg.index_add_(0, rows, w)
    return mass, deg


def row_kl(
    empirical_mass: torch.Tensor, predicted: torch.Tensor, degree: torch.Tensor
) -> torch.Tensor:
    """Sum of $\\mathrm{KL}(\\hat h \\Vert \\tilde h)$ over rows with ``degree > 0``."""
    present = degree > 0
    if not bool(present.any()):
        return predicted.new_zeros(())
    emp = empirical_mass[present]
    pred = predicted[present]
    emp = emp / emp.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    pred = pred / pred.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return (emp * (log_clamp(emp) - log_clamp(pred))).sum()


def partner_kl_terms(
    q_batch: torch.Tensor,
    q_logits: torch.Tensor,
    prob_kk: torch.Tensor,
    adj_out: csr_matrix,
    adj_in: csr_matrix,
    idx: np.ndarray,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Out- and in-neighbour partner KLs for the minibatch nodes ``idx``.

    ``adj_out`` / ``adj_in`` are CSR with the same row index as the full graph
    (``adj_in`` is the transpose of ``adj_out``). Empirical mixes use the full
    graph (minus held-out positives), not the induced minibatch block.
    ``prob_kk[k,k'] = P(edge | z_i=k, z_j=k')``. Predicted out-mix is $q\\,P$;
    predicted in-mix is $q\\,P^\\top$.
    """
    mass_out, deg_out = neighbor_type_mass(adj_out[idx], q_logits, device, dtype)
    mass_in, deg_in = neighbor_type_mass(adj_in[idx], q_logits, device, dtype)
    pred_out = q_batch @ prob_kk
    pred_in = q_batch @ prob_kk.T
    kl_out = row_kl(mass_out, pred_out, deg_out)
    kl_in = row_kl(mass_in, pred_in, deg_in)
    return kl_out, kl_in
