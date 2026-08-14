"""Unit tests for partner-histogram consistency (not homophily)."""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix

from partner_consistency import (
    binary_adjacency_without_heldout,
    neighbor_type_mass,
    partner_kl_terms,
    row_kl,
)


def test_heldout_positives_are_stripped() -> None:
    adj = csr_matrix(np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]], dtype=np.float32))
    held = {
        "src": np.array([0, 1]),
        "tgt": np.array([1, 2]),
        "y": np.array([1.0, 0.0]),  # only (0,1) is a held-out positive
    }
    out = binary_adjacency_without_heldout(adj, held)
    assert out[0, 1] == 0.0
    assert out[0, 2] == 1.0
    assert out[1, 2] == 1.0  # negative held-out pair was never an edge


def test_neighbor_mass_is_stopgrad() -> None:
    # 0 -> 1, node 0's empirical mix should be q_1, with no grad into q_1.
    adj = csr_matrix(np.array([[0, 1], [0, 0]], dtype=np.float32))
    q_logits = torch.tensor([[0.0, 4.0], [4.0, 0.0]], requires_grad=True)
    mass, deg = neighbor_type_mass(adj[[0]], q_logits, "cpu", torch.float32)
    assert deg[0].item() == 1.0
    q1 = torch.softmax(q_logits[1].detach(), dim=-1)
    assert torch.allclose(mass[0], q1)
    assert not mass.requires_grad


def test_row_kl_zero_when_match() -> None:
    h = torch.tensor([[0.1, 0.9], [0.5, 0.5]])
    deg = torch.tensor([2.0, 1.0])
    assert float(row_kl(h, h.clone(), deg)) < 1e-6


def test_bipartite_partners_not_homophily() -> None:
    """Type-0 nodes connect to type-1; matching h to q (homophily) would be wrong."""
    # Nodes 0,1 type 0; nodes 2,3 type 1. Edges 0,1 -> 2,3.
    a = np.zeros((4, 4), dtype=np.float32)
    a[0, 2] = a[0, 3] = a[1, 2] = a[1, 3] = 1.0
    adj = csr_matrix(a)
    q_logits = torch.tensor(
        [[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]],
        dtype=torch.float32,
    )
    q = torch.softmax(q_logits, dim=-1)
    # Correct block: type 0 -> type 1.
    eta_good = torch.tensor([[-8.0, 8.0], [-8.0, -8.0]])
    eta_homophily = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
    p_good = torch.sigmoid(eta_good)
    p_bad = torch.sigmoid(eta_homophily)
    idx = np.array([0, 1, 2, 3])
    kl_good_out, _ = partner_kl_terms(
        q, q_logits, p_good, adj, adj.T.tocsr(), idx, "cpu", torch.float32
    )
    kl_bad_out, _ = partner_kl_terms(
        q, q_logits, p_bad, adj, adj.T.tocsr(), idx, "cpu", torch.float32
    )
    assert float(kl_good_out) < 0.05, kl_good_out
    assert float(kl_bad_out) > float(kl_good_out) + 0.5, (kl_bad_out, kl_good_out)


if __name__ == "__main__":
    test_heldout_positives_are_stripped()
    test_neighbor_mass_is_stopgrad()
    test_row_kl_zero_when_match()
    test_bipartite_partners_not_homophily()
    print("partner_consistency tests ok")
