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
   - LV / LV+$e$ / GNN: ~50k held-out positive directed edges + 50k negatives. Held-out
     positives are removed from the GNN message-passing graph and excluded from the
     training LL.
   - PCA: ~10% of rows held out for reconstruction scoring.
2. **Phase 1 — HP search:** each method sweeps its own grid; pick the setting that
   maximizes held-out Bernoulli log-likelihood (LV / LV+$e$ / GNN) or minimizes
   held-out row MSE (PCA; stored as negated MSE so higher is always better).
3. **Phase 2 — multi-seed finals:** retrain the selected setting with several seeds
   (default `0,1,2`). Report Hungarian / ARI / NMI vs visual types as mean ± std.

Orchestration: [`run_experiments.py`](run_experiments.py). Outputs:
`hp_results.*`, `hp_best.json`, `final_results.*`, `final_summary.*`.
Select methods with `--methods` (e.g. `pca lv lv_e` to skip GNN).

Lean LV+$e$ HP grid: $d\in\{32,64\}$, $d_e\in\{16,32\}$, $\mathrm{lr}\in\{0.05,0.1\}$,
$e_{\mathrm{wd}}\in\{10^{-3},10^{-2}\}$. Optional GNN grid: $L\in\{0,1,2,4\}$,
$d\in\{32,64\}$, $\mathrm{lr}\in\{0.005,0.01\}$.


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
`final_summary.csv` in the working directory.

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
  --methods pca lv lv_e \
  --detach-only \
  --remote-stop-after
```


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
