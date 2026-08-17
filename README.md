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

The main setting is the **full brain**: every method is fitted on the complete
$n=134\,181$ connectome and scored on the labelled subset. A narrower
visual-system scope is available as an alternative protocol and is reported
separately in [Visual-system subgraph protocol](#visual-system-subgraph-protocol-alternative-scope);
the two scopes are different fitting problems and their numbers are not
interchangeable.

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

### Full-brain sink protocol

Steps 1–3 score a partition only on the $46\,479$ graph nodes carrying a visual
type, which silently grants the method the visual / non-visual split — an
annotation no unsupervised method receives — and says nothing about what it
does with the remaining $87\,702$ neurons. `build_nonvisual_sink_gt`
([`evaluate_clustering.py`](evaluate_clustering.py)) therefore extends the
ground truth to all $134\,181$ graph nodes by giving every unlabelled node one
shared sink label `__nonvisual__`. Nodes a predictor never scored are collected
into a single reserved cluster rather than dropped, so fitting on a subgraph is
not silently rewarded. [`reeval_nonvisual_sink.py`](reeval_nonvisual_sink.py)
re-scores assignment dictionaries already on disk under both protocols; no
model is retrained.

```bash
uv run python reeval_nonvisual_sink.py --pred reeval_artifacts/*_assignment_dict.npy
```

| partition | fitted on | sink Hungarian | fraction | sink ARI | sink NMI | visual fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| single-cluster floor | — | $87\,702$ | $65.4\%$ | $0$ | $0$ | — |
| split-only oracle | — | $90\,209$ | $67.2\%$ | $0.766$ | $0.434$ | $5.4\%$ |
| LV-vSBM headline ($\lambda=1$) | full brain | $28\,936\pm66$ | $21.6\%$ | $0.005$ | $0.492$ | $59.5\%$ |
| LV-vSBM ($\lambda=0$) | full brain | $28\,279\pm355$ | $21.1\%$ | $0.005$ | $0.471$ | $57.7\%$ |
| PCA $+$ $k$-means | full brain | $2\,075$ | $1.5\%$ | $0.000$ | $0.104$ | $3.1\%$ |
| **unseeded NTAC (HP stage, seed $0$)** | full brain | $\mathbf{38\,579}$ | $\mathbf{28.8\%}$ | $0.023$ | $0.460$ | $65.2\%$ |
| unseeded NTAC | visual subgraph | $122\,653\pm913$ | $91.4\%$ | $0.999$ | $0.929$ | $75.2\%$ |

Fractions are of the Hungarian maximum on the node set concerned. The LV-vSBM
and visual-subgraph NTAC rows average three seeds; the full-brain NTAC row is a
single run, for the reasons given below.

The headline configuration drops from $59.5\%$ on the visual-only protocol to
$21.6\%$ here, *below* the trivial single-cluster floor of $65.4\%$. It covers
the whole graph but re-uses its $729$ clusters for the non-visual brain, so its
visual-only score was measuring type recovery **given** the split rather than
recovery of cell-type structure from connectivity alone.

The last row is not a competing number and the two NTAC rows are not
comparable. That run was fitted on the visual subgraph
(see [Visual-system subgraph protocol](#visual-system-subgraph-protocol-alternative-scope)),
so $87\,702$ of its nodes are unscored and
collapse into one reserved cluster that happens to align almost perfectly with
the sink; roughly $65$ percentage points of its $91.4\%$ are that free
alignment, which the split-only oracle row isolates. A visual-scope partition
must not be ranked against full-brain methods under this protocol.

**Provenance of the full-brain NTAC row.** The historical full-brain unseeded
NTAC baseline ($30\,654.5\pm24.7$ on the visual-only protocol, two seeds) left
no per-neuron assignment dictionaries and therefore could not be re-scored under
the sink protocol. A fresh full-brain run at $K=729$, $R=12$, $T=0.1$ on
`CPU_X_16` (studio `flywrite-full-ntac`; unseeded NTAC does not repay a GPU, see
[Device selection](#device-selection)) supplied the missing assignments. Only
the hyperparameter stage was retained: multi-seed finals were **not** run, since
a single equitable partition of the whole graph already answers the question the
sink protocol asks, and the historical seed-to-seed spread
($\pm24.7$ on $30\,654.5$, i.e. $\approx0.08\%$) is two orders of magnitude
below the effects at issue. That the single run reproduces the historical
visual-only score to within $1.2\%$ ($30\,282$ against $30\,654.5$ at identical
$K$, $R$ and $T$) is consistent with the row being representative rather than a
seed artefact, though with $n=1$ this cannot be asserted with a confidence
interval.

```bash
# re-score the retained assignment dictionary under both protocols
uv run python reeval_nonvisual_sink.py \
  --pred full_ntac_results/hp_ntac_k729_R12_T0.1_assignment_dict.npy
```

**Interpretation.** NTAC leads LV-vSBM under both protocols — $28.8\%$ against
$21.6\%$ on the sink, $65.2\%$ against $59.5\%$ on the visual-only subset — so
the ranking established in [Results](#results) survives the change of scoring
scope. What does not survive is the standing of either method against the
trivial baseline: at $28.8\%$, NTAC also falls far below the $65.4\%$
single-cluster floor. Its $396$ recovered clusters are distributed over the
whole brain rather than concentrated on the visual system, so the sink is
fragmented much as LV's is, and a sink ARI of $0.023$ against the split-only
oracle's $0.766$ indicates that essentially none of the visual / non-visual
boundary is recovered. The reading given above for LV-vSBM thus extends to
NTAC: a visual-only score of $65.2\%$ quantifies type recovery **conditional on**
the visual / non-visual split, not recovery of cell-type structure from
connectivity alone.

Baselines kept in-tree: PCA $+$ $k$-means
([`train_pca_baseline.py`](train_pca_baseline.py)) and unseeded NTAC
([`train_ntac.py`](train_ntac.py)).

## Visual-system subgraph protocol (alternative scope)

This section describes an **optional alternative** to the full-brain setting
above, not a replacement for it. Here both NTAC and LV-vSBM are fitted on the
induced subgraph of the $46\,479$ neurons carrying a FlyWire visual-type
annotation, so that training scope and evaluation scope coincide. Its purpose
is a like-for-like comparison against the NTAC paper, which works in the
visual system: on the full brain a method must also spend capacity on the
$87\,702$ unlabelled neurons, and one may reasonably object that this
handicaps it relative to the published setting. The subgraph protocol removes
that objection at the cost of conditioning the whole experiment on the
visual / non-visual split — information no unsupervised method is otherwise
given. For that reason the full brain remains the primary scientific setting
and these numbers are read alongside, not against, the table above.

Export the graph once, then select it by scope:

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

### LV hyperparameter grid on the visual subgraph

The induced graph is a different optimisation problem from the full brain, so
the full-brain optimum is not transferable and the visual scope is searched on
its own $36$-point grid: $d\in\{256,512\}$, learning rate
$\in\{0.01,0.03\}$, entropy weight $\beta_H\in\{0.1,0.3,1.0\}$ and
partner-histogram KL weight $\lambda\in\{0,0.3,1\}$, with `bfs_frac=1`, a
Bernoulli likelihood and unit-norm cluster embeddings at fixed scale $16$ held
fixed. Selection uses held-out edge AUC at $40$ epochs — the same budget as the
finals, since a shorter search ranks configurations by their transient early
behaviour rather than by where they converge:

```bash
uv run python launch_lightning_sweep.py \
  --studio-name flywrite-visual-lv-hp --machine T4 \
  --graph-scope visual --methods lv --phase all \
  --lv-dims 256 512 --lv-lrs 0.01 0.03 --lv-bfs-fracs 1.0 \
  --lv-likelihoods bernoulli \
  --lv-u-norms unit --lv-u-scales fixed --lv-u-scale-inits 16.0 \
  --lv-entropy-betas 0.1 0.3 1.0 --lv-partner-kl-weights 0 0.3 1.0 \
  --select-metric auc --grad-clip 1.0 \
  --epochs 40 --final-epochs 40 --final-seeds 0 1 2 \
  --detach-only --remote-stop-after
```

### Results under the alternative scope

Both methods are fitted and scored on the same $46\,479$ vertices, three seeds
each, with the LV configuration selected by the grid above ($d=256$, learning
rate $0.03$, $\beta_H=0.3$, $\lambda=1$).

| method (visual scope) | Hungarian | fraction | ARI | NMI |
| --- | ---: | ---: | ---: | ---: |
| Unseeded NTAC | $\mathbf{34\,951 \pm 913}$ | $\mathbf{75.2\%}$ | $0.754$ | $0.902$ |
| LV-vSBM | $15\,182 \pm 194$ | $32.7\%$ | $0.255$ | $0.819$ |

Restricting the graph moves the two methods in opposite directions. NTAC
improves on its full-brain score ($30\,654$ to $34\,951$), whereas LV-vSBM
falls well below its own ($27\,644$ to $15\,182$) even after its
hyperparameters are re-searched on this scope. Inducing on visual neurons
deletes every edge to the rest of the brain, and LV-vSBM estimates block
structure from exactly those connectivity profiles, so the truncation removes
evidence it relies on while leaving NTAC's equitable partitioning of the
labelled vertex set intact. Whether the gap is intrinsic or an artefact of a
$36$-point grid is not settled by these runs; both readings are consistent with
the near-identical held-out AUC ($0.987$) across the grid's top configurations.

Partitions produced under this scope cover only the $46\,479$ labelled vertices,
so they cannot be entered into the
[full-brain sink protocol](#full-brain-sink-protocol) as competitors: there,
every unscored node collapses into a reserved cluster that aligns with the sink
and inflates the score by roughly $65$ percentage points.

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
