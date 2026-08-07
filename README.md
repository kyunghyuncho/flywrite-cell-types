# Clustering of Neurons from the Fruit Fly Connectome

This repository implements variational stochastic block models (vSBMs) for clustering
neurons in the [FlyWire](https://codex.flywire.ai/) connectome according to
**inter-cluster** connectivity patterns, rather than shared individual neighbours.

A detailed derivation of the single-hop low-rank vSBM appears in
[Stochastic variational inference for low-rank stochastic block models](https://kyunghyuncho.me/stochastic-variational-inference-for-low-rank-stochastic-block-models-or-how-i-re-discovered-sbm-unnecessarily/).
This repository additionally provides a **GNN-augmented** decoder that incorporates
multi-hop soft assignment context.

## Data

Unfiltered FlyWire connections (data version 783):

* https://codex.flywire.ai/api/download?data_product=connections_no_threshold&data_version=783
* https://codex.flywire.ai/api/download?data_product=names&data_version=783

Build the sparse binary adjacency and label dictionaries:

```bash
uv venv
source .venv/bin/activate
uv sync --exclude-newer "1 week"
uv run python ./connectivity_matrix_construction.py
uv run python ./visual_neuron_type_dict.py
```

## Models

### Low-rank vSBM (single-hop baseline)

Implemented in [`hidden_markov_graph.py`](hidden_markov_graph.py). Each neuron $i$ has a
mean-field approximate posterior $\alpha_i=\mathrm{softmax}(\beta_i)$ over $K$ clusters.
Directed edge probabilities depend only on the latent clusters of the endpoints:

$$
p(e_{ij}=1\mid z_i,z_j)=\sigma\!\big(u_s^{z_i}\cdot u_t^{z_j}+b\big).
$$

Parameters $\{\beta,U_s,U_t,b\}$ are trained by minibatch maximization of the variational
lower bound (Bernoulli likelihood under $q$ plus entropy of $\alpha$).

### GNN-vSBM (multi-hop residual)

Implemented in [`gnn_vsbm.py`](gnn_vsbm.py). Soft assignments feed a **low-rank LV
bilinear backbone**, and a directed GNN supplies a residual multi-hop correction:

$$
\mathrm{logit}_{ij}
=
(\alpha_i U_s)\cdot(\alpha_j U_t)+b
+\gamma\,(h_i^{\mathrm{src}}\cdot h_j^{\mathrm{tgt}}).
$$

Node features start at $h_i^{(0)}=\sum_k\alpha_i^k u_k$ and are refined by $L$ directed
GNN layers on a minibatch-edge-masked adjacency (in/out aggregation, LayerNorm,
residual). **Jumping Knowledge** concatenates $[h^{(0)},\ldots,h^{(L)}]$ and projects
back to $d$ before the src/tgt heads, so deeper layers expand the receptive field
without erasing 0-hop cluster identity. Learnable $\gamma$ is initialized at $0$, so
$L{=}0$ recovers an LV-like decoder.

On this connectome, undirected hop balls (median, excl. self) are already large:
$L{=}1\sim 18$, $L{=}2\sim 3\times 10^3$, $L{=}3\sim 3.7\times 10^4$,
$L{=}4\sim 10^5$ nodes. Minibatch edge masking removes $\ll 1\%$ of edges, so
neighbourhood starvation is not the issue—oversmoothing from replacing the LV
decoder was. The residual+JK design addresses that.

Smoke / short run:

```bash
uv run python gnn_vsbm.py --epochs 1 --max-updates 5 --minibatch 1024 --seed 0 --layers 2
```

Longer training (defaults: $K=729$, $d=32$, $L=2$):

```bash
uv run python gnn_vsbm.py --epochs 20 --minibatch 2048 --lr 0.01 --seed 0 --out-prefix gnn
```

### PCA + $k$-means baseline

Implemented in [`sparse_graph_pca.py`](sparse_graph_pca.py): stochastic linear
autoencoding of the adjacency followed by $k$-means in the embedding space.

## Evaluation protocol (unsupervised selection)

Ground-truth visual types are **not** used for hyperparameter selection. Selection uses
held-out unsupervised metrics only ([`heldout.py`](heldout.py)):

1. **Fixed splits** (shared across methods; `split_seed=0` by default):
   - LV / GNN: ~50k held-out positive directed edges + 50k negatives. Held-out positives
     are removed from the GNN message-passing graph and excluded from the training LL.
   - PCA: ~10% of rows held out for reconstruction scoring.
2. **Phase 1 — HP search:** each method sweeps its own grid; pick the setting that
   maximizes held-out Bernoulli log-likelihood (LV/GNN) or minimizes held-out row MSE
   (PCA; stored as negated MSE so higher is always better).
3. **Phase 2 — multi-seed finals:** retrain the selected setting with several seeds
   (default `0..4`). Report Hungarian / ARI / NMI vs visual types as mean ± std.

Orchestration: [`run_experiments.py`](run_experiments.py). Outputs:
`hp_results.*`, `hp_best.json`, `final_results.*`, `final_summary.*`.

```bash
# Local (CPU/MPS; slow for full grids)
uv run python run_experiments.py --device cpu --phase all

# Lightning L4 (detached; Studio stops itself when done)
source ~/.ortet/lightning.env
export LIGHTNING_USERNAME=kc119 LIGHTNING_TEAMSPACE=vision-model
uv run python launch_lightning_sweep.py --machine L4 --detach-only --remote-stop-after
```

Lean default GNN HP grid: $L\in\{0,1,2,4\}$, $d\in\{32,64\}$,
$\mathrm{lr}\in\{0.005,0.01\}$ (short HP epochs, full finals, 3 seeds).

Single-run evaluation against GT (after training):

```bash
uv run python evaluate_clustering.py \
  --pred final_gnn_*_assignment_dict.npy \
         final_lv_*_assignment_dict.npy \
         final_pca_*_assignment_dict.npy
```

Interactive inspection:
[`inspect_sweep_results.ipynb`](inspect_sweep_results.ipynb).

### Earlier GT-selected sweep (for reference only)

An earlier L4 grid selected GNN configs by Hungarian vs GT (flawed for unsupervised
methods). Under that protocol the best GNN scored **3078**, LV **2717**, PCA **1440**
(see historical [`sweep_results.csv`](sweep_results.csv)). Those numbers are **not**
comparable to the unsupervised protocol above.


## Environment

This project uses [`uv`](https://github.com/astral-sh/uv). Dependencies are declared in
[`pyproject.toml`](pyproject.toml). Prefer:

```bash
uv run python <script>.py
```

over ad-hoc `pip install`. A legacy [`requirements.txt`](requirements.txt) is retained
for reference but is no longer the primary install path.

## Training on Lightning AI (recommended for full runs)

Full-graph GNN training on CPU/MPS is slow ($\sim$1–2 s/step). Use a Lightning Studio
GPU (L4 or T4) via [`launch_lightning_train.py`](launch_lightning_train.py):

```bash
# Programmatic keys: lightning.ai → profile → Keys
source ~/.ortet/lightning.env   # or export LIGHTNING_USER_ID / LIGHTNING_API_KEY
export LIGHTNING_USERNAME=kc119
export LIGHTNING_TEAMSPACE=vision-model

uv pip install lightning-sdk
uv run python launch_lightning_train.py --machine L4 --epochs 20 --stop-after
```

**Note.** GPU machines require a verified payment method on Lightning. After billing is
enabled, the launcher boots an L4, optionally uploads code/data, trains with
`--device cuda`, evaluates, downloads artifacts, and can stop the Studio with
`--stop-after`. Use `--skip-upload` when the Studio already has the files.


## License

MIT. See [`LICENSE`](LICENSE).
