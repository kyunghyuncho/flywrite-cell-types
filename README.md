# Clustering Neurons in the FlyWire Connectome

This repository recovers cell-type structure in the
[FlyWire](https://codex.flywire.ai/) adult fly brain from **synaptic
connectivity alone**. The model is a low-rank variational stochastic block
model (LV-vSBM). Ground-truth visual types ($K=729$ labelled neurons among
$n_{\mathrm{shared}}=46\,479$) are used only for reporting; hyperparameter
selection never sees those labels.

A derivation of the generative model and the variational bound is in
[Stochastic variational inference for low-rank stochastic block models](https://kyunghyuncho.me/stochastic-variational-inference-for-low-rank-stochastic-block-models-or-how-i-re-discovered-sbm-unnecessarily/).
What follows is the training recipe that makes that model competitive on
FlyWire, and how it compares to the connectivity-only baseline NTAC
(Schwartzman et al., *Nat Commun* 2026).

## Results

Agreement with the $729$ FlyWire visual types on the labelled subset.
Hyperparameters are chosen by held-out edge AUC; Hungarian / ARI / NMI /
$K_{\mathrm{pred}}$ are diagnostics. The theoretical maximum Hungarian score
is $46\,479$; a random $K=729$ assignment scores $\approx 980$.

| method | Hungarian | fraction | ARI | NMI | $K_{\mathrm{pred}}$ | seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unseeded NTAC | $30\,654.5 \pm 24.7$ | $66.0\%$ | $0.676$ | $0.878$ | $\approx 226$–$260$ | $2$ |
| **LV-vSBM (headline)** | $\mathbf{27\,644 \pm 32}$ | $\mathbf{59.5\%}$ | $0.600$ | $0.889$ | $362\pm15$ | $3$ |
| PCA $+$ $k$-means | $1\,355.0 \pm 88.8$ | $2.9\%$ | $0.008$ | $0.225$ | — | $3$ |

The headline LV configuration reaches about $90\%$ of NTAC’s Hungarian score
(gap $\approx 3\,010$ neurons). NTAC remains coarser in $K_{\mathrm{pred}}$;
LV sits between NTAC’s resolution and the labelled type count of $729$.

**Headline hyperparameters.** $d=512$, learning rate $0.03$, Bernoulli
likelihood, BFS minibatches with `bfs_frac=1`, unit-norm cluster embeddings
with fixed scale $16$, entropy weight $\beta_H=0.3$, partner-histogram KL
weight $\lambda=1$, $40$ epochs, seeds $0/1/2$. Held-out AUC
$0.9897\pm0.0002$.

## Model

Each neuron $i$ carries a mean-field posterior $q_i\in\Delta^{K-1}$ over $K$
clusters. Directed edge probabilities depend only on the latent clusters of
the two endpoints. With unit-norm cluster embeddings $\hat u_k,\hat v_{k'}$
and a scalar scale $s$,

$$
p(e_{ij}=1\mid z_i=k,\,z_j=k')
=\sigma\!\bigl(s\cdot\hat u_k^\top\hat v_{k'}+b\bigr)
=\sigma(\eta_{kk'}).
$$

This is a rank-$d$ parameterization of the $K\times K$ block-logit table
$\eta$. Training maximises a minibatch ELBO: an exact Bernoulli (or Poisson /
negative-binomial) reconstruction term on the sampled subgraph, minus an
entropy regulariser $\beta_H\sum_i H(q_i)$, plus an optional partner-consistency
term described below. Implementation:
[`hidden_markov_graph.py`](hidden_markov_graph.py),
[`train_lv_vsbm.py`](train_lv_vsbm.py).

## Training design

Four choices define the method. None of them adds decoder capacity; each
constrains how the existing bilinear SBM sees the graph and how soft
assignments are regularised.

### 1. Neighbourhood minibatches

Uniform rectangular blocks $A[I,J]$ on independent node sets almost never
contain edges at FlyWire density ($\approx 1.5\times 10^{-4}$). The trainer
instead grows a node set by BFS on $A+A^\top$ and scores the **square induced
subgraph** ([`subgraph_sampler.py`](subgraph_sampler.py)). With
`bfs_frac=1`, a coverage-defined epoch observes nearly the full edge set. An
epoch is one full pass over neurons, not a fixed step count.

### 2. Bounded block logits

Without a norm constraint, the bilinear decoder saturates and training
collapses. Cluster embeddings are $\ell_2$-normalised per row
(`--u-norm unit`) with a fixed scalar scale (`--u-scale fixed
--u-scale-init 16`). Learning rates are kept at or below $0.03$; gradient
clipping and best-validation checkpointing
([`training_utils.py`](training_utils.py)) keep saturated states from being
reported as solutions.

### 3. Entropy weight $\beta_H$

The minibatch ELBO includes $\beta_H\sum_{i\in\mathrm{batch}} H(q_i)$ (sum
entropy, not a batch mean). The default $\beta_H=1$ is the uniform-prior
bound; AUC selection on FlyWire prefers $\beta_H=0.3$. Selection never uses
the reweighted training objective itself—only held-out edge metrics.

### 4. Partner-histogram consistency ($\lambda$)

FlyWire visual types are defined by *who a neuron connects to*, not by
resembling its neighbours (homophily). Let $A$ be the binary adjacency with
held-out positive pairs removed, and write $h_i^{\mathrm{out}}$ for the
empirical out-neighbour type mix (stop-gradient). The model predicts the
partner mix of $i$’s own soft type as $\tilde h_i=q_i\,\sigma(\eta)$. The
regulariser is

$$
\lambda\Bigl(
\mathrm{KL}\!\bigl(h_i^{\mathrm{out}}\,\big\|\,\tilde h_i\bigr)
+\mathrm{KL}\!\bigl(h_i^{\mathrm{in}}\,\big\|\,q_i\,\sigma(\eta)^\top\bigr)
\Bigr),
$$

implemented in [`partner_consistency.py`](partner_consistency.py). Empirically,
**$\lambda$ should be capped at $1$**: larger weights continue to raise
held-out AUC while Hungarian agreement collapses (a proxy/target split). The
headline uses $\lambda=1$; $\lambda=0$ is the no-op control.

What we deliberately *do not* use: per-neuron residual embeddings, GNNs over
$q$ or over residuals, or adjacency smoothness that pulls neighbouring
posteriors together. Those improve edge calibration and hurt type recovery.
If block flexibility is revisited, prefer a free $K\times K$ logit table over
an MLP decoder.

## Evaluation protocol

1. Hold out a balanced set of directed pairs ([`heldout.py`](heldout.py)).
2. Sweep hyperparameters; rank by held-out AUC (`--select-metric auc`).
3. Retrain the selected configuration for several seeds; report mean $\pm$ std
   of Hungarian, ARI, and NMI against FlyWire types on the labelled neurons.

Orchestration: [`run_experiments.py`](run_experiments.py). Remote sweeps:
[`launch_lightning_sweep.py`](launch_lightning_sweep.py). Inspection notebook:
[`inspect_sweep_results.ipynb`](inspect_sweep_results.ipynb).

Baselines kept in-tree: PCA $+$ $k$-means
([`train_pca_baseline.py`](train_pca_baseline.py)) and unseeded NTAC
([`train_ntac.py`](train_ntac.py)).

## Visual-system subgraph protocol

Protocol (1) fits both NTAC and LV-vSBM on the induced subgraph of neurons with
FlyWire visual-type annotations. This matches the NTAC paper's visual-system
setting more closely than fitting the full-brain graph and restricting only the
reported assignments. Export the graph once, then select it by scope:

```bash
uv run python export_visual_subgraph.py \
  --adjacency sparse_connectivity_matrix.npz \
  --mapping root_id_to_index_mapping.json \
  --visual-types visual_neuron_types.csv.gz

uv run python run_experiments.py \
  --graph-scope visual --methods ntac lv \
  --ntac-max-ks 729
```

The same files may instead be supplied explicitly with `--adjacency`,
`--mapping`, `--heldout-pairs`, and `--heldout-rows`. For the OL-intrinsic
ablation, export with `--category "OL intrinsic"` and pass its
`*_ol_intrinsic` adjacency and mapping explicitly.

### Isolated vertices under NTAC

Inducing on visual neurons deletes every cross-region edge, which strands $24$
of the $46\,479$ vertices with no surviving in-graph partner. Upstream
`ntac.unseeded.convert.problem_from_data` builds its vertex-name list from edge
endpoints alone and then indexes it with the original matrix indices, so any
isolate shifts the two out of registration and aborts the run with an
`IndexError`. [`train_ntac.py`](train_ntac.py) therefore drops degree-zero
vertices before handing the graph to NTAC and restores them afterwards under a
dedicated residual label: equitable partitioning has no evidence about them, and
folding them into a real cluster would corrupt that cluster's Hungarian match.
NTAC is thus still scored on exactly the vertex set LV and PCA are scored on.

### Device selection

Unseeded NTAC offloads only the weighted-Jaccard distance kernel, and it
re-uploads the whole CSR on every call, so the GPU buys little. Measured on the
visual subgraph to $k=32$: $48.0$ s on an L4 against $61.9$ s on $16$ vCPU,
$49.7$ s on $8$ and $66.8$ s on $4$ — about $1.3$–$1.9\times$, far below the GPU
price premium. NTAC accordingly runs on a CPU Studio via
`launch_lightning_sweep.py --device cpu`, while LV keeps the GPU. The wrapper
prints the distance kernel it actually selected, because NTAC reverts to CPU
silently when the numba CUDA toolchain fails to link.

## Setup

**Data** (FlyWire data version 783, unfiltered connections and names):

* https://codex.flywire.ai/api/download?data_product=connections_no_threshold&data_version=783
* https://codex.flywire.ai/api/download?data_product=names&data_version=783

```bash
uv venv && source .venv/bin/activate
uv sync --exclude-newer "1 week"
uv run python ./connectivity_matrix_construction.py
uv run python ./visual_neuron_type_dict.py
```

**Headline LV run** (local GPU):

```bash
uv run python run_experiments.py \
  --device cuda --phase all --methods lv \
  --lv-dims 512 --lv-lrs 0.03 \
  --lv-bfs-fracs 1.0 --lv-likelihoods bernoulli \
  --lv-u-norms unit --lv-u-scales fixed --lv-u-scale-inits 16.0 \
  --lv-entropy-betas 0.3 \
  --lv-partner-kl-weights 1 \
  --select-metric auc --grad-clip 1.0 \
  --epochs 40 --final-epochs 40 --final-seeds 0 1 2
```

**Lean exploratory grid** (smaller $d$, no partner KL):

```bash
uv run python run_experiments.py \
  --device cuda --phase all --methods lv \
  --lv-dims 64 128 256 \
  --lv-lrs 0.003 0.01 0.03 \
  --lv-bfs-fracs 1.0 --lv-likelihoods bernoulli \
  --lv-u-norms unit --lv-u-scales fixed --lv-u-scale-inits 12.0 16.0 \
  --select-metric auc --grad-clip 1.0 \
  --epochs 40 --final-epochs 40 --final-seeds 0 1 2
```

**Lightning** (T4 is enough for LV):

```bash
export LIGHTNING_USERNAME=kc119 LIGHTNING_TEAMSPACE=vision-model
# credentials: source ~/.ortet/lightning.env
uv run python launch_lightning_sweep.py \
  --machine T4 --methods lv \
  --lv-dims 512 --lv-lrs 0.03 \
  --lv-u-norms unit --lv-u-scales fixed --lv-u-scale-inits 16.0 \
  --lv-entropy-betas 0.3 \
  --lv-partner-kl-weights 1 \
  --lv-likelihoods bernoulli --lv-bfs-fracs 1.0 \
  --select-metric auc --grad-clip 1.0 \
  --epochs 40 --final-epochs 40 --final-seeds 0 1 2 \
  --detach-only --remote-stop-after
```

Use `--skip-existing` to resume a Studio without wiping artefacts. When sweeping
$\beta_H$ or $\lambda$, ensure `hp_best.json` propagates `entropy_beta` and
`partner_kl_weight` into finals (otherwise retrains silently fall back to
defaults $1$ and $0$).

Dependencies are listed in `pyproject.toml` / `requirements.txt` (PyTorch,
SciPy, scikit-learn, `ntac`, Lightning SDK for remote sweeps).

## License

See the repository license file.
