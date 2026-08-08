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

### Subgraph minibatches and count likelihoods (LV and LV+$e$)

Both LV trainers ([`train_lv_vsbm.py`](train_lv_vsbm.py),
[`train_lv_e.py`](train_lv_e.py)) share the sampler in
[`subgraph_sampler.py`](subgraph_sampler.py) and support weighted edge
likelihoods. The GNN variants are unchanged.

**Why.** The original minibatch drew two *independent* node permutations $I$ and
$J$ and scored the rectangular block $A[I,J]$. At density $1.5\times10^{-4}$ a
$2048\times2048$ block contains only $\approx 625$ of the $2.7\times10^{6}$
edges, so a $20$-epoch run observed roughly $0.3\times$ the edge set — the model
essentially never saw the graph, only its sparsity.

**Square induced subgraphs.** A single node set $S$ now indexes both rows and
columns, and $S$ is grown by breadth-first expansion over the undirected pattern
$A+A^{\top}$ (CSR `indptr`/`indices`, vectorized frontier gathers; sampling costs
$\ll1$ s per epoch). Reciprocal edges and within-neighbourhood structure are
therefore present in the block. The graph has no self-loops, so the diagonal is
masked out of the loss exactly like the held-out pairs.

**Mixing (`--bfs-frac`).** A pure BFS block is roughly two orders of magnitude
denser than the graph, which inflates the learned base rate $b$ and destroys
calibration against the held-out pairs. Each batch therefore mixes a
`--bfs-frac` fraction of BFS-grown nodes with a uniformly sampled remainder:
$0$ recovers the previous uniform sampling, $1$ gives a pure neighbourhood
subgraph, trainer default $0.5$. In the pilot the mixing was monotone in the
wrong direction for calibration but the right one for recovery, and $1$ won
outright, so the sweep pins it there rather than spending a grid axis on it.

Note that the held-out Bernoulli log-likelihood does **not** arbitrate this on its
own. The held-out set is balanced ($50$k positives, $50$k negatives) while the
graph is $1.5\times10^{-4}$ dense, so a model whose base rate has been inflated by
training on dense blocks scores *better* there regardless of what it learned. The
trainers therefore also report `val_auc`, the held-out ROC AUC, which is invariant
to any monotone recalibration and so measures ranking of edges above non-edges
directly. The two can disagree sharply: at `bfs_frac=0.5` over three epochs the
log-likelihood degraded monotonically ($-2.80\to-3.08$) while AUC improved
($0.68\to0.79$), so early stopping on log-likelihood would have selected the worst
of the three models.

**Coverage.** An epoch is one full pass over the node set: BFS seeds and the
uniform component are both drawn from a pool of nodes not yet visited in that
epoch, BFS prefers unvisited candidates and re-anchors on a fresh seed when an
expansion mostly recycles seen nodes, and the epoch ends when the pool empties.
Each epoch prints a greppable diagnostic,

```
[coverage] epoch=0 batches=90 bfs_frac=0.5 visits_min=1 visits_median=1.0 \
  visits_mean=1.36 visits_max=60 uncovered=0 pos_edge_obs=524964 \
  cum_pos_edge_obs=524964 edges_seen_x_nnz=0.19
```

Measured at $|S|=2048$ on the FlyWire graph (one epoch, full coverage in every
case):

| `bfs_frac` | batches/epoch | edges per block | edge observations / epoch | $\times$ nnz |
|---|---|---|---|---|
| 0.0 | 66 | 625 | 41 208 | 0.02 |
| 0.25 | 74 | 3 289 | 243 357 | 0.09 |
| 0.5 | 90 | 5 835 | 524 964 | 0.19 |
| 1.0 | 226 | 11 318 | 2 557 158 | 0.95 |

Because an epoch is defined by coverage rather than by a fixed batch count, its
cost grows with `bfs_frac` ($66$ batches at $0$ versus $226$ at $1$), so the
epoch-matched grid also hands the BFS arms $3.4\times$ more gradient updates. To
separate the sampler from the extra compute, `--target-updates` sets a total
update budget that overrides `--epochs`, and `run_experiments.py
--lv-control-updates N` adds a matched-compute control at `bfs_frac=0`. Trainer
defaults were raised to
`--epochs 40` so the edge set is traversed several times;
[`run_experiments.py`](run_experiments.py) still overrides with `--epochs` /
`--final-epochs`.

**Count likelihoods (`--likelihood {bernoulli,poisson,nb}`).** Synapse counts run
from $1$ to $18$ (mean $1.43$; $73\%$ equal to $1$) and binarization discards
them. For `poisson` and `nb` the adjacency is *not* binarized and the existing
linear predictor $\eta$ is read as a log-rate,

$$
\log p_{\mathrm{Poi}}(y)=y\eta-e^{\eta}-\log y!,
\qquad
\log p_{\mathrm{NB}}(y)=\log\frac{\Gamma(y+r)}{\Gamma(r)\,y!}
+r\log\frac{r}{r+\mu}+y\log\frac{\mu}{r+\mu},
$$

with $\mu=e^{\eta}$ and a learnable scalar dispersion parameterized as $\log r$;
$\eta$ is clamped to $[-30,10]$. `train_lv_vsbm.py` never forms per-pair logits,
so the count terms are folded into its exact $K\times K$ aggregation: with
$w^{\mathrm{count}}=q^{\top}(C\odot M)q$ and $w^{\mathrm{all}}=q^{\top}Mq$ for
keep-mask $M$, the Poisson objective is
$\sum_{kk'}[w^{\mathrm{count}}_{kk'}\eta_{kk'}-w^{\mathrm{all}}_{kk'}e^{\eta_{kk'}}]$,
and for the negative binomial the two $\log(r/(r+\mu))$ and $\log(\mu/(r+\mu))$
terms aggregate the same way while $\log\Gamma(y+r)-\log\Gamma(r)$ is evaluated
directly on the block. `train_lv_e.py` scores pairs directly and simply swaps its
per-pair term. Terms constant in the parameters ($-\log y!$) are dropped from the
training objective and retained in the held-out report.

**Selection stays Bernoulli.** Log-likelihoods of different families are not
comparable, so `val_metric` remains the held-out mean Bernoulli log-likelihood
(`val_metric_name = heldout_bernoulli_ll`) for every variant, comparable with all
earlier `lv` / `lv_e` results. For count models the implied edge probability
$P(y>0)=1-e^{-\lambda}$ (Poisson) or $1-(r/(r+\mu))^{r}$ (NB) is converted to a
logit and scored against the binary held-out labels. The native count
log-likelihood is reported separately as `val_native_ll` / `val_native_ll_name`;
held-out counts are read from the unbinarized adjacency, which is a label lookup
rather than leakage because those pairs are masked out of training.

### Low-rank vSBM + node residuals $e_i$ (no GNN)

Implemented in [`train_lv_e.py`](train_lv_e.py). Same LV bilinear backbone, plus a
per-node residual embedding $e_i\in\mathbb{R}^{d_e}$:

$$
\mathrm{logit}_{ij}
=
(\alpha_i U_s)\cdot(\alpha_j U_t)+b
+e_i\cdot e_j.
$$

Adam applies weight decay **only** to $e$ (`--e-wd`) so residuals stay small and do
not absorb cluster identity; clusters remain $\arg\max_k\alpha_i^k$.

Smoke:

```bash
uv run python train_lv_e.py --epochs 1 --max-updates 5 --minibatch 1024 --seed 0
uv run python train_lv_e.py --epochs 1 --max-updates 5 --bfs-frac 1.0 --likelihood nb --seed 0
```

### GNN$_e$-vSBM (multi-hop residual over $e_i$)

Implemented in [`gnn_e_vsbm.py`](gnn_e_vsbm.py). LV bilinear backbone is unchanged;
the directed GNN now message-passes **node residuals** $e_i$ rather than cluster
features $\alpha U$:

$$
\mathrm{logit}_{ij}
=
(\alpha_i U_s)\cdot(\alpha_j U_t)+b
+\gamma\,(h_i^{\mathrm{src}}\cdot h_j^{\mathrm{tgt}}),
\qquad
h^{(0)}=e.
$$

Jumping Knowledge and residual $\gamma$ (init $0$) match the GNN-vSBM design.
Weight decay applies only to raw $e$.

```bash
uv run python gnn_e_vsbm.py --epochs 1 --max-updates 5 --layers 2 --seed 0
```

### Unseeded NTAC (connectivity-only equitable partitioning)

Wraps the official implementation of
[Schwartzman et al., Nat Commun 2026](https://www.nature.com/articles/s41467-025-68044-1)
([`ntac`](https://github.com/BenJourdan/ntac)). Unseeded NTAC grows an approximate
equitable partition using seeded NTAC as a subroutine. Paper defaults:
$K{=}729$, $R{=}12$ seeded iterations, $T{=}0.1$ seed-candidate fraction.
Selection uses negated mean Jaccard cost (higher better).

```bash
uv run python train_ntac.py --max-k 729 --max-iterations 12 --frac-seeds 0.1 --seed 0
```

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
   - LV / LV+$e$ / GNN / GNN$_e$: ~50k held-out positive directed edges + 50k negatives.
     Held-out positives are removed from the GNN message-passing graph and excluded
     from the training LL.
   - PCA: ~10% of rows held out for reconstruction scoring.
2. **Phase 1 — HP search:** each method sweeps its own grid; pick the setting that
   maximizes held-out Bernoulli log-likelihood (LV / LV+$e$ / GNN / GNN$_e$) or
   minimizes held-out row MSE (PCA; stored as negated MSE so higher is always better).
   Count-likelihood LV variants are also selected on the held-out **Bernoulli**
   log-likelihood implied by the fitted count model, so `val_metric` stays on one
   scale; their native log-likelihood is recorded as `val_native_ll`.
3. **Phase 2 — multi-seed finals:** retrain the selected setting with several seeds
   (default `0,1,2`). Report Hungarian / ARI / NMI vs visual types as mean ± std.

Orchestration: [`run_experiments.py`](run_experiments.py). Outputs:
`hp_results.*`, `hp_best.json`, `final_results.*`, `final_summary.*`.
Select methods with `--methods` (e.g. `lv lv_e ntac`).

Current LV grids fix the learning rate at the previously selected optimum and
spend the budget on the likelihood and the rank $d$. A pilot at $5$ epochs
established that `bfs_frac` is not worth a grid axis — Hungarian $1\,228$ at $0$,
$1\,644$ at $0.5$, $4\,641$ at $1$ — so the sampler is pinned at
`bfs_frac`$=1$ and $d$ is swept instead, on the reasoning that once data exposure
is no longer binding the rank bottleneck ($d=64$ describing $729$ types) becomes
the next constraint. Sweeping $d$ is nearly free because the $K\times K$ term
dominates at fixed $K=729$.

* **LV** (`--lv-lrs 0.1 --lv-bfs-fracs 1.0`): `--lv-dims 64 128 256` $\times$
  `--lv-likelihoods bernoulli poisson nb` — $9$ runs.
* **Matched-compute control** (`--lv-control-updates N`, off by default): one
  extra LV run at `bfs_frac`$=0$ with a total budget of $N$ updates, at the first
  entry of `--lv-dims` and `--lv-likelihoods` only. It reproduces the previous
  uniform-sampling configuration at the compute of a BFS arm; in the pilot it
  reached $2\,257$ against $4\,641$, so the sampler helps beyond the extra
  gradient steps.
* **LV+$e$** (`--lv-e-d-es 16 --lv-e-lrs 0.05 --lv-e-wds 0.01
  --lv-e-bfs-fracs 1.0`): `--lv-e-dims 64 128` $\times$
  `--lv-e-likelihoods bernoulli poisson` — $4$ runs.

That is $14$ HP runs with the control enabled, plus $3\times2=6$ multi-seed
finals. `--bfs-seeds` (default $4$) is shared and not swept. Lean GNN$_e$ grid:
$L\in\{0,1,2\}$,
$d\in\{32,64\}$, $d_e{=}16$, $\mathrm{lr}\in\{0.005,0.01\}$, $e_{\mathrm{wd}}{=}10^{-2}$.
NTAC defaults: $K{=}729$, $R{=}12$, $T{=}0.1$ (paper). Optional α-GNN grid:
$L\in\{0,1,2,4\}$, $d\in\{32,64\}$, $\mathrm{lr}\in\{0.005,0.01\}$.


## End-to-end workflow

Prerequisites: data artifacts in the repo root
(`sparse_connectivity_matrix.npz`, `root_id_to_index_mapping.json`,
`root_id_type_dict.pkl`) and a `uv` environment (`uv sync --exclude-newer "1 week"`).

### Local (CPU / MPS / local CUDA)

Full unsupervised HP search + multi-seed finals (slow on CPU/MPS):

```bash
uv run python run_experiments.py --device cpu --phase all
# or: --device mps / --device cuda
```

Useful flags: `--phase hp|final`, `--skip-existing`, `--epochs` / `--final-epochs`,
`--gnn-layers 0 1 2 4`, `--final-seeds 0 1 2`.

When finished, inspect results:

```bash
# tabular summary
uv run python -c "import pandas as pd; print(pd.read_csv('final_summary.csv').to_string(index=False))"

# Hungarian / ARI / NMI for saved assignment dicts
uv run python evaluate_clustering.py \
  --pred final_gnn_*_assignment_dict.npy \
         final_lv_*_assignment_dict.npy \
         final_pca_*_assignment_dict.npy

# interactive plots (HP leaderboard, val vs GT, seed uncertainty)
uv run jupyter notebook inspect_sweep_results.ipynb
```

The notebook expects `hp_results.csv`, `hp_best.json`, `final_results.csv`, and
`final_summary.csv` in the working directory. Selection metrics are method-specific
(held-out Bernoulli LL, negated row MSE, negated mean Jaccard cost), so `val_metric`
is only ever plotted on per-method axes; only Hungarian / ARI / NMI are comparable
across methods.

#### Recovering an interrupted sweep

If a remote sweep is cut short after the HP phase, the finals rows can be
reconstructed offline from the saved assignment dictionaries — the ground-truth
metrics are a cheap CPU computation and do not require retraining. For the
NTAC-only sweep (whose artifacts were retrieved with an `ntac_` prefix):

```bash
uv run python merge_ntac_results.py --dry-run   # inspect the planned merge
uv run python merge_ntac_results.py             # write, backing up to *.prentac.bak
```

The script recomputes Hungarian / ARI / NMI with `evaluate_clustering.evaluate_pair`,
takes the unsupervised `val_metric` from the surviving remote log, and merges the
rows into `hp_results.*`, `hp_best.json`, `final_results.*`, and `final_summary.*`
using the same sorted key-union schema as `run_experiments.py`. It is idempotent:
existing `ntac` rows are replaced rather than duplicated.

Single-model smoke tests (optional):

```bash
uv run python train_pca_baseline.py --max-iter 200 --seed 0 --out-prefix smoke_pca
uv run python train_lv_vsbm.py --epochs 1 --max-updates 5 --seed 0 --out-prefix smoke_lv
uv run python gnn_vsbm.py --epochs 1 --max-updates 5 --layers 2 --seed 0 --out-prefix smoke_gnn
```

### Lightning AI (recommended for the full grid)

Install the SDK once, and put API keys in `~/.ortet/lightning.env`
(`LIGHTNING_USER_ID`, `LIGHTNING_API_KEY`). GPU Studios need a verified payment method.

**1. Launch a detached sweep** (uploads code/data, starts `run_experiments.py` under
`nohup`, returns immediately). By default the Studio **stops itself** when the job
finishes, so a closed laptop still ends billing:

```bash
source ~/.ortet/lightning.env
export LIGHTNING_USERNAME=kc119 LIGHTNING_TEAMSPACE=vision-model
uv pip install lightning-sdk

uv run python launch_lightning_sweep.py \
  --machine L4 \
  --methods lv lv_e \
  --detach-only \
  --remote-stop-after
```

The launcher now defaults to `--methods lv lv_e` with `--epochs 15
--final-epochs 40`, matching the sampler / likelihood grid above; pass
`--methods pca lv lv_e gnn_e` to restore the previous full sweep.

**Auto-sleep.** The Studio ships with idle auto-sleep enabled at the platform
default, and a detached `nohup` sweep does not reliably register as activity: an
earlier NTAC sweep was stopped mid-run, losing a seed and the `final_summary`
artifacts. The launcher therefore sets `studio.auto_sleep = False` before
starting the machine and aborts if the setting does not take, since a silent kill
several hours in is far more expensive than a failed launch. Billing still ends
on completion because `--remote-stop-after` stops the Studio from inside the
remote job on any exit code; the residual exposure is a Studio left running if
that wrapper itself is killed, so check the Studio state after a sweep that ends
abnormally. Pass `--keep-auto-sleep` to opt out.


Omit `--detach-only` to poll from the client and download artifacts when done
(requires the laptop to stay online). Use `--skip-upload` on restarts if the Studio
already has code and data.

**2. Check progress** while the Studio is running:

```bash
uv run python -c "
import os
from lightning_sdk import Studio
s = Studio('flywrite-gnn-vsbm',
           teamspace=os.environ['LIGHTNING_TEAMSPACE'],
           user=os.environ['LIGHTNING_USERNAME'])
print(s.run_with_exit_code('bash /teamspace/studios/this_studio/remote_status_unsup_sweep.sh')[0])
"
```

Remote log: `/teamspace/studios/this_studio/unsup_sweep.log`. Done marker:
`unsup_sweep.done` (contains the exit code).

**3. Download results and inspect locally.** If the Studio already auto-stopped,
start a cheap CPU instance only for download, then stop again:

```bash
source ~/.ortet/lightning.env
export LIGHTNING_USERNAME=kc119 LIGHTNING_TEAMSPACE=vision-model

uv run python - <<'PY'
import os
from pathlib import Path
from lightning_sdk import Studio, Machine

s = Studio(
    "flywrite-gnn-vsbm",
    teamspace=os.environ["LIGHTNING_TEAMSPACE"],
    user=os.environ["LIGHTNING_USERNAME"],
)
s.start(Machine.CPU)
root = Path(".")
for name in [
    "hp_best.json",
    "hp_results.csv",
    "final_summary.csv",
    "final_results.csv",
    "final_summary.json",
    "final_results.json",
    "unsup_sweep.log",
]:
    try:
        s.download_file(name, str(root / name))
        print("<-", name)
    except Exception as e:
        print("!", name, e)

out, _ = s.run_with_exit_code(
    "cd /teamspace/studios/this_studio && "
    "python -c \"import glob; print('\\\\n'.join("
    "sorted(glob.glob('final_*_assignment_dict*.npy'))))\""
)
for line in (out or "").splitlines():
    name = line.strip()
    if name.endswith(".npy"):
        s.download_file(name, str(root / name))
        print("<-", name)
s.stop()
PY

uv run jupyter notebook inspect_sweep_results.ipynb
```

Alternatively, a single-run GNN train (no HP sweep) is available via
[`launch_lightning_train.py`](launch_lightning_train.py):

```bash
uv run python launch_lightning_train.py --machine L4 --epochs 20 --stop-after
```


## Environment

This project uses [`uv`](https://github.com/astral-sh/uv). Dependencies are declared in
[`pyproject.toml`](pyproject.toml). Prefer:

```bash
uv run python <script>.py
```

over ad-hoc `pip install`. A legacy [`requirements.txt`](requirements.txt) is retained
for reference but is no longer the primary install path.


## License

MIT. See [`LICENSE`](LICENSE).
