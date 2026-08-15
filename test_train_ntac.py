"""Regression tests for the unseeded-NTAC wrapper.

The induced visual subgraph strands a few degree-zero vertices, which used to
abort the whole sweep inside ``ntac.unseeded.convert.problem_from_data``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from train_ntac import _compact_nonisolated, _patch_ntac_cuda


def _matrix_with_interior_isolate() -> sparse.csr_matrix:
    """Four vertices, one edge ``0 <-> 3``; vertices 1 and 2 are isolates.

    The isolates sit *below* an edge endpoint, so the truncated upstream name
    list (two entries) is indexed with 3.
    """
    return sparse.csr_matrix(
        (
            np.ones(2, dtype=np.float32),
            (np.array([0, 3]), np.array([3, 0])),
        ),
        shape=(4, 4),
    )


def test_compact_nonisolated_drops_isolates_and_keeps_index_map() -> None:
    compact, active_indices = _compact_nonisolated(_matrix_with_interior_isolate())

    np.testing.assert_array_equal(active_indices, np.array([0, 3]))
    np.testing.assert_array_equal(
        compact.toarray(),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )


def test_compact_nonisolated_keeps_vertices_with_only_incoming_edges() -> None:
    """The graph is directed, so an in-edge alone must count as non-isolated."""
    adjacency = sparse.csr_matrix(
        (np.ones(1, dtype=np.float32), (np.array([0]), np.array([2]))),
        shape=(3, 3),
    )

    _, active_indices = _compact_nonisolated(adjacency)

    np.testing.assert_array_equal(active_indices, np.array([0, 2]))


def test_uncompacted_isolates_break_upstream_converter() -> None:
    """Pin the upstream bug this wrapper works around.

    ``problem_from_data`` derives vertex names from edge endpoints only, then
    indexes them with the original matrix indices.
    """
    _patch_ntac_cuda(force_cpu=True)
    from ntac import GraphData
    from ntac.unseeded import convert

    adjacency = _matrix_with_interior_isolate()
    data = GraphData(adjacency, labels=np.array(["0"] * 4, dtype=object))

    with pytest.raises(IndexError):
        convert.problem_from_data(data)


def test_compacted_graph_converts_consistently() -> None:
    """After compaction the converter agrees with the matrix it is handed."""
    _patch_ntac_cuda(force_cpu=True)
    from ntac import GraphData
    from ntac.unseeded import convert

    compact, _ = _compact_nonisolated(_matrix_with_interior_isolate())
    data = GraphData(compact, labels=np.array(["0", "0"], dtype=object))

    problem = convert.problem_from_data(data)

    assert problem.numv == compact.shape[0]
    np.testing.assert_array_equal(problem.A().toarray(), compact.toarray())
