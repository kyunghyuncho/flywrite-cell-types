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

### GNN-vSBM (multi-hop)

Implemented in [`gnn_vsbm.py`](gnn_vsbm.py). Soft assignments are projected into a
$d$-dimensional space, $h_i^{(0)}=\sum_k\alpha_i^k u_k$, and refined by $L$ directed
GNN layers that aggregate incoming and outgoing neighbours on a
**minibatch-edge-masked** adjacency:

$$
h^{(l+1)}=\phi\Big(W_{\mathrm{self}}h^{(l)}+W_{\mathrm{in}}\tilde{A}^\top h^{(l)}+W_{\mathrm{out}}\tilde{A}h^{(l)}\Big).
$$

Source/target heads then decode

$$
p(e_{ij}=1)=\sigma\!\big((h_i^{\mathrm{src}})\cdot(h_j^{\mathrm{tgt}})+b\big).
$$

Edge masking removes directed edges whose both endpoints lie in the current minibatch
so the decoder cannot trivially read the labels it is asked to reconstruct. The same
ELBO is maximized with Adam.

Smoke / short run:

```bash
uv run python gnn_vsbm.py --epochs 1 --max-updates 5 --minibatch 1024 --seed 0
```

Longer training (defaults: $K=729$, $d=32$, $L=2$):

```bash
uv run python gnn_vsbm.py --epochs 20 --minibatch 2048 --lr 0.01 --seed 0 --out-prefix gnn
```

### PCA + $k$-means baseline

Implemented in [`sparse_graph_pca.py`](sparse_graph_pca.py): stochastic linear
autoencoding of the adjacency followed by $k$-means in the embedding space.

## Evaluation

Cluster quality is measured by aligning predicted clusters to the 729 annotated visual
neuron types with the Hungarian algorithm on the confusion matrix (higher is better),
plus ARI and NMI. See [`evaluate_clustering.py`](evaluate_clustering.py):

```bash
uv run python evaluate_clustering.py \
  --pred gnn_assignment_dict_729.npy \
         cluster_assignment_dict_729.npy \
         pca_cluster_assignment_dict_729.npy
```

Interactive exploration remains available in [`cluster_similarity_test.ipynb`](cluster_similarity_test.ipynb).

### Reference scores (visual neurons, $K=729$, seed $0$, $20$ epochs on L4)

Hyperparameter grid for GNN-vSBM: $L\in\{1,2,3\}$, $d\in\{32,64\}$,
$\mathrm{lr}\in\{0.005,0.01,0.05\}$. Full table in [`sweep_results.csv`](sweep_results.csv).

| Method | Hungarian vs GT | Notes |
| --- | ---: | --- |
| **GNN-vSBM best** ($L{=}1$, $d{=}64$, $\mathrm{lr}{=}0.005$) | **3078** | best of 18 GNN configs |
| GNN-vSBM ($L{=}1$, $d{=}32$, $\mathrm{lr}{=}0.01$) | 2990 | close second |
| Low-rank vSBM (re-run, $20$ epochs) | 2717 | single-hop baseline on same budget |
| Low-rank vSBM (earlier long run) | 3139 | saved historical checkpoint |
| PCA + $k$-means (re-run) | 1440 | $d{=}32$, $10^4$ SGD steps |
| Random (optimistic / pessimistic) | $\approx$1230 / $\approx$980 | |

**Takeaways.** On a matched $20$-epoch budget, the best GNN decoder (**3078**) beats the
re-run single-hop vSBM (**2717**) and PCA (**1440**). Shallower GNNs with smaller learning
rates worked best; $\mathrm{lr}{=}0.05$ consistently underperformed. Deeper stacks ($L{=}2,3$)
did not help under this training budget—an open question for longer runs or residual /
normalization ablations.

Reproduce the sweep:

```bash
source ~/.ortet/lightning.env
export LIGHTNING_USERNAME=kc119 LIGHTNING_TEAMSPACE=vision-model
uv run python launch_lightning_sweep.py --machine L4 --epochs 20 --seed 0 --stop-after
```


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
