"""Square induced-subgraph minibatches for the low-rank vSBM trainers.

The original LV minibatch drew two *independent* node permutations and scored the
rectangular block $A[I, J]$. At a density of $1.5\\times10^{-4}$ such a block of
size $2048\\times2048$ contains only $\\approx 629$ edges, so a full training run
observed a small fraction of the $2.7$M edges. Here a single node set $S$ indexes
both rows and columns, and $S$ is grown by breadth-first expansion over the
undirected pattern $A+A^{\\top}$, which concentrates edges inside the block.

A pure BFS block is far denser than the graph, which biases the learned base rate
upwards and destroys calibration against the held-out pairs. The batch node set is
therefore a mixture: a fraction ``bfs_frac`` of the budget comes from BFS
expansion and the remainder is drawn uniformly at random. ``bfs_frac=0``
reproduces uniform sampling (up to the block now being square) and ``bfs_frac=1``
gives a pure neighbourhood subgraph.

Coverage within an epoch is guaranteed by a pool of nodes not yet visited in that
epoch: BFS seeds and the uniform component are both drawn from the pool, BFS
prefers unvisited candidates when expanding, and the epoch ends once the pool is
exhausted.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.sparse import csr_matrix


def symmetrized_csr(adj: csr_matrix) -> csr_matrix:
    """Unweighted CSR sparsity pattern of $A+A^{\\top}$ used for BFS expansion."""
    pattern = csr_matrix(
        (np.ones(adj.nnz, dtype=np.int8), adj.indices.copy(), adj.indptr.copy()),
        shape=adj.shape,
    )
    sym = (pattern + pattern.T).tocsr()
    sym.sort_indices()
    return sym


class SubgraphBatchSampler:
    """Yield node sets for square induced-subgraph minibatches.

    Parameters
    ----------
    adj:
        Directed adjacency; only its sparsity pattern is used.
    batch_size:
        Node budget per batch (block is ``batch_size x batch_size``).
    bfs_frac:
        Fraction of the budget filled by BFS expansion; the rest is uniform.
    n_seeds:
        Number of BFS seeds per expansion round.
    seed:
        RNG seed.
    coverage_factor:
        Safety cap on batches per epoch, as a multiple of ``ceil(n / batch_size)``.
    """

    def __init__(
        self,
        adj: csr_matrix,
        batch_size: int,
        bfs_frac: float = 0.5,
        n_seeds: int = 4,
        seed: int = 0,
        coverage_factor: float = 4.0,
    ) -> None:
        if not 0.0 <= bfs_frac <= 1.0:
            raise ValueError(f"bfs_frac must lie in [0, 1], got {bfs_frac}")
        self.n = int(adj.shape[0])
        self.batch_size = int(min(batch_size, self.n))
        self.bfs_frac = float(bfs_frac)
        self.n_seeds = max(1, int(n_seeds))
        self.coverage_factor = float(coverage_factor)
        self.rng = np.random.default_rng(seed)

        if self.bfs_frac > 0.0:
            sym = symmetrized_csr(adj)
            self.indptr = sym.indptr.astype(np.int64, copy=False)
            self.indices = sym.indices.astype(np.int64, copy=False)
        else:
            self.indptr = np.zeros(self.n + 1, dtype=np.int64)
            self.indices = np.zeros(0, dtype=np.int64)

        self._in_batch = np.zeros(self.n, dtype=bool)
        self._visited = np.zeros(self.n, dtype=bool)
        self._visits = np.zeros(self.n, dtype=np.int64)
        self._pool = np.zeros(0, dtype=np.int64)
        self._cursor = 0

    @property
    def visit_counts(self) -> np.ndarray:
        """Per-node visit counts accumulated over the current epoch."""
        return self._visits

    def _neighbors(self, frontier: np.ndarray) -> np.ndarray:
        starts = self.indptr[frontier]
        counts = self.indptr[frontier + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        base = np.repeat(starts, counts)
        ramp = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        return self.indices[base + ramp]

    def _take_pool(self, k: int) -> np.ndarray:
        """Pop up to ``k`` not-yet-visited nodes from the epoch pool."""
        if k <= 0:
            return np.zeros(0, dtype=np.int64)
        taken: list[np.ndarray] = []
        need = k
        while need > 0 and self._cursor < self._pool.size:
            end = min(self._cursor + max(2 * need, 1024), self._pool.size)
            chunk = self._pool[self._cursor : end]
            offsets = np.nonzero(~self._visited[chunk])[0]
            if offsets.size == 0:
                self._cursor = end
                continue
            use = offsets[:need]
            taken.append(chunk[use])
            need -= use.size
            self._cursor += int(use[-1]) + 1
        if not taken:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(taken)

    def _random_seeds(self, k: int) -> np.ndarray:
        cand = self.rng.integers(0, self.n, size=4 * k, dtype=np.int64)
        cand = np.unique(cand)
        cand = cand[~self._in_batch[cand]]
        return cand[:k]

    def _bfs_nodes(self, budget: int) -> np.ndarray:
        """Grow up to ``budget`` nodes by BFS, reseeding when a component runs out."""
        out = np.empty(budget, dtype=np.int64)
        n_out = 0

        def admit(nodes: np.ndarray) -> np.ndarray:
            nonlocal n_out
            if nodes.size == 0:
                return nodes
            nodes = nodes[~self._in_batch[nodes]][: budget - n_out]
            if nodes.size == 0:
                return nodes
            self._in_batch[nodes] = True
            out[n_out : n_out + nodes.size] = nodes
            n_out += nodes.size
            return nodes

        frontier = np.zeros(0, dtype=np.int64)
        while n_out < budget:
            if frontier.size == 0:
                seeds = self._take_pool(self.n_seeds)
                seeds = seeds[~self._in_batch[seeds]]
                if seeds.size == 0:
                    seeds = self._random_seeds(self.n_seeds)
                frontier = admit(seeds)
                if frontier.size == 0:
                    break
                continue

            remaining = budget - n_out
            if frontier.size > remaining:
                frontier = self.rng.choice(frontier, size=remaining, replace=False)
            cand = np.unique(self._neighbors(frontier))
            if cand.size:
                cand = cand[~self._in_batch[cand]]
            if cand.size == 0:
                frontier = np.zeros(0, dtype=np.int64)
                continue
            # Prefer nodes not yet seen this epoch so coverage keeps advancing.
            fresh = cand[~self._visited[cand]]
            stale = cand[self._visited[cand]]
            self.rng.shuffle(fresh)
            self.rng.shuffle(stale)
            admitted = admit(np.concatenate([fresh, stale]))
            # An expansion that mostly recycles already-seen nodes is a dead end
            # for coverage: keep the nodes but re-anchor on an unvisited seed.
            n_fresh = min(int(fresh.size), int(admitted.size))
            frontier = admitted if 2 * n_fresh >= admitted.size else np.zeros(0, dtype=np.int64)
        return out[:n_out]

    def epoch(self) -> Iterator[np.ndarray]:
        """Iterate over one epoch of node sets, covering every node at least once."""
        self._visited[:] = False
        self._visits[:] = 0
        self._pool = self.rng.permutation(self.n).astype(np.int64, copy=False)
        self._cursor = 0

        n_bfs = int(round(self.batch_size * self.bfs_frac))
        n_rand = self.batch_size - n_bfs
        max_batches = int(np.ceil(self.coverage_factor * self.n / self.batch_size))

        for _ in range(max_batches):
            parts: list[np.ndarray] = []
            if n_bfs > 0:
                parts.append(self._bfs_nodes(n_bfs))
            if n_rand > 0:
                rand = self._take_pool(n_rand)
                rand = rand[~self._in_batch[rand]]
                self._in_batch[rand] = True
                parts.append(rand)
            batch = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
            self._in_batch[batch] = False
            if batch.size < 2:
                break
            self._visits[batch] += 1
            self._visited[batch] = True
            yield np.sort(batch)
            if self._cursor >= self._pool.size and not self._visited.all():
                # Pool drained while stragglers remain: reseed from those.
                left = np.nonzero(~self._visited)[0]
                self._pool = self.rng.permutation(left)
                self._cursor = 0
            if self._visited.all():
                break

    def coverage_line(
        self,
        epoch: int,
        n_batches: int,
        pos_obs: int,
        cum_pos_obs: int,
        nnz: int,
    ) -> str:
        """Greppable per-epoch coverage diagnostic (prefix ``[coverage]``)."""
        v = self._visits
        return (
            f"[coverage] epoch={epoch} batches={n_batches} bfs_frac={self.bfs_frac} "
            f"visits_min={int(v.min())} visits_median={float(np.median(v)):.1f} "
            f"visits_mean={float(v.mean()):.2f} visits_max={int(v.max())} "
            f"uncovered={int((v == 0).sum())} "
            f"pos_edge_obs={pos_obs} cum_pos_edge_obs={cum_pos_obs} "
            f"edges_seen_x_nnz={cum_pos_obs / max(nnz, 1):.2f}"
        )
