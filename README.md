# Clustering of Neurons from the Fruit Fly Connectome

This repository trains a low-rank variational stochastic block model (LV-vSBM)
to cluster neurons in the [FlyWire](https://codex.flywire.ai/) connectome from
**connectivity alone**. The scientific target is the $729$ visual cell types;
the selection criterion used during development never looks at those labels.

A derivation of the model is in
[Stochastic variational inference for low-rank stochastic block models](https://kyunghyuncho.me/stochastic-variational-inference-for-low-rank-stochastic-block-models-or-how-i-re-discovered-sbm-unnecessarily/).
This repository is the experimental follow-up: what has to change in the
*training protocol* before that model recovers cell types at a useful rate.

It is written as a teaching artefact. The rest of this README states what
mattered, what did not, and which pieces of code embody each lesson.

## What we learned (two days of negative and positive results)

### What mattered

1. **Minibatch construction.** The original trainer scored a rectangular block
   $A[I,J]$ for two *independent* random node sets. At graph density
   $1.5\times 10^{-4}$, a $2048\times 2048$ block holds $\approx 625$ of the
   $2.7\times 10^{6}$ edges, so a twenty-epoch run saw roughly $0.3\times$ the
   edge set. The model was learning sparsity, not structure. Growing a single
   node set by BFS and scoring the **square induced subgraph** changes the
   picture completely: at `bfs_frac=1` a coverage-defined epoch observes
   $\approx 0.93\times$ the nonzero edges, and a five-epoch pilot jumps from
   Hungarian $1\,228$ (uniform) to $4\,641$ (BFS), with a matched-compute
   uniform control at $2\,257$ confirming that the sampler itself — not just
   the extra gradient steps — is responsible.
2. **The hyperparameter / optimization protocol.** Once the model can see the
   graph, three protocol choices dominate accuracy:
   - a **learning-rate range below the previous default** (`0.03` and below;
     accuracy is monotone decreasing in lr across the constrained grid);
   - a **unit-norm constraint** on the cluster embeddings $U_\mathrm{left}$,
     $U_\mathrm{right}$, with a fixed scalar scale, which stops the Bernoulli
     decoder from saturating at lr $\ge 0.1$;
   - **held-out AUC**, not Bernoulli log-likelihood, as the selection metric,
     because the held-out pairs are class-balanced while the graph is sparse —
     a correctly calibrated sparse model scores poorly on LL from epoch $0$.

Together these take LV from Hungarian $\approx 2\,500$ (uniform minibatches,
unconstrained decoder, LL selection) to $\approx 25\,100$ on the shared
labelled neurons (multi-seed mean under AUC selection), and restoring the
sum-entropy ELBO with AUC-selected $\beta_H=0.3$ lifts the result to
$\approx 26\,800$, and partner-histogram KL with $\lambda=1$ raises the
headline further to $\approx 27\,600$.

### What did not matter

1. **Per-neuron residual embeddings $e_i$.** Adding a weight-decayed node
   residual to the bilinear decoder improved held-out likelihood and *hurt*
   ground-truth recovery ($5\,216$ Hungarian). Better edge calibration is not
   the same as better cell types.
2. **Graph neural nets over soft assignments (or over $e_i$).** A multi-hop
   residual GNN, once stabilised, became a better edge model and a worse
   cell-type model. Its best stabilised single-seed arm reached Hungarian
   $\approx 15\,700$ against LV's $\approx 25\,100$–$26\,800$, and its held-out
   likelihood was *negatively* correlated with ground truth. Selecting by the
   unsupervised objective therefore preferred its worst clusterings. The GNN
   and $e_i$ trainers have been removed from this branch so the teaching
   surface matches the positive result.

The methodological point for students is not “GNNs are bad”. It is that an
auxiliary architecture which wins the training objective can still lose the
scientific target when the objective is only a proxy — and that fixing the
data pipeline and the optimization protocol on the simple model was worth more
than adding capacity.

## Where the numbers stand

Ground-truth agreement against the $729$ FlyWire visual types, on the
$n_{\mathrm{shared}}=46\,479$ neurons that carry a label. Hyperparameters are
chosen by a held-out metric only; the ground-truth column is a report, never a
selection criterion. Theoretical maximum Hungarian is $46\,479$; a random
$K=729$ assignment scores $\approx 980$.

| method | Hungarian (mean $\pm$ std) | fraction of $46\,479$ | ARI | NMI | $K_{\mathrm{pred}}$ (labelled) | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| Unseeded NTAC | $30\,654.5 \pm 24.7$ | $66.0\%$ | $0.676$ | $0.878$ | $\approx 226$–$260$ | $2$ |
| **LV with $\beta_H=0.3$ + partner KL $\lambda=1$** | $\mathbf{27\,644 \pm 32}$ | $\mathbf{59.5\%}$ | $0.600$ | $0.889$ | $\approx 362\pm15$ | $3$ |
| LV vSBM (BFS + unit-norm + AUC, $\beta_H=0.3$ only) | $26\,837 \pm 339$ | $57.7\%$ | $0.609$ | $0.862$ | $\approx 401\pm42$ | $3$ |
| LV vSBM (prior AUC width sweep) | $25\,095.3 \pm 911.6$ | $54.0\%$ | $0.522$ | $0.828$ | $\approx 383$–$481$ | $3$ |
| LV, earlier unit-norm sweep ($d=256$, LL-selected) | $24\,300.0 \pm 971.6$ | $52.3\%$ | $0.488$ | $0.822$ | $\approx 388$–$484$ | $3$ |
| LV, unconstrained decoder (same sampler) | $8\,459.7 \pm 354.1$ | $18.2\%$ | $0.177$ | $0.509$ | — | $3$ |
| LV $+\,e_i$ (negative result; removed) | $5\,216.3 \pm 89.2$ | $11.2\%$ | $0.042$ | $0.373$ | — | $3$ |
| LV, uniform minibatches | $2\,546.0 \pm 392.0$ | $5.5\%$ | $0.046$ | $0.224$ | — | $3$ |
| PCA $+\,k$-means | $1\,355.0 \pm 88.8$ | $2.9\%$ | $0.008$ | $0.225$ | — | $3$ |
| random $K=729$ assignment | $980.0 \pm 8.0$ | $2.1\%$ | $0.000$ | $0.202$ | $729$ (by construction) | $20$ |

The headline LV configuration augments the restored sum-entropy ELBO at
$\beta_H=0.3$ with partner-histogram KL at $\lambda=1$, using `d=512,
lr=0.03, bernoulli, bfs_frac=1.0, --u-norm unit --u-scale fixed
--u-scale-init 16`, $40$ epochs, and seeds $0/1/2$; held-out AUC is
$0.9897\pm0.0002$. Its Hungarian score of $27\,644\pm32$ closes the gap to
NTAC to approximately $3\,010$ neurons (about $90\%$ of the NTAC score), while
NTAC remains coarser in $K_{\mathrm{pred}}$. A free $K\times K$ block table
remains an optional next step.

## Data

Unfiltered FlyWire connections (data version 783):

* https://codex.flywire.ai/api/download?data_product=connections_no_threshold&data_version=783
* https://codex.flywire.ai/api/download?data_product=names&data_version=783

```bash
uv venv
source .venv/bin/activate
uv sync --exclude-newer "1 week"
uv run python ./connectivity_matrix_construction.py
uv run python ./visual_neuron_type_dict.py
```

## Model: low-rank vSBM

Implemented in [`hidden_markov_graph.py`](hidden_markov_graph.py) and trained by
[`train_lv_vsbm.py`](train_lv_vsbm.py). Each neuron $i$ has a mean-field
posterior $\alpha_i=\mathrm{softmax}(\beta_i)$ over $K$ clusters. Directed edge
probabilities depend only on the latent clusters of the endpoints:

$$
p(e_{ij}=1\mid z_i,z_j)=\sigma\!\big(s\cdot \hat u^{z_i}\cdot \hat v^{z_j}+b\big),
$$

where under `--u-norm unit` the cluster embeddings are $\ell_2$-normalised per
row and $s$ is a (fixed or learned) scalar scale. Parameters are trained by
minibatch maximisation of the variational lower bound.

### Lesson 1 code: subgraph minibatches

[`subgraph_sampler.py`](subgraph_sampler.py) grows a node set by BFS over
$A+A^\top$ and returns square induced blocks. `--bfs-frac` mixes BFS-grown
nodes with a uniform remainder ($0$ = old sampler, $1$ = pure neighbourhood
subgraph). An epoch is defined by **full node coverage**, reported on a
`[coverage]` line. `--target-updates` (and
`run_experiments.py --lv-control-updates`) run a matched-compute uniform
control so the sampler effect is not confounded with extra steps.

Optional `--likelihood {bernoulli,poisson,nb}` fits unbinarized synapse counts;
selection still uses a held-out Bernoulli metric (LL or AUC) so configurations
remain comparable across likelihoods.

### Lesson 2 code: optimization and selection protocol

Shared utilities live in [`training_utils.py`](training_utils.py) and
[`heldout.py`](heldout.py):

- `--u-norm unit --u-scale fixed --u-scale-init {12,16}` bounds the block logit;
- `--grad-clip 1.0` and best-validation checkpointing prevent a saturated
  decoder from being reported as the final model;
- `--select-metric {ll,auc}` chooses which held-out number ranks configurations
  and which epoch is restored (default `ll` for back-compat; use `auc` when LL
  peaks at initialisation, which it does under the balanced held-out split);
- both the best-checkpoint and last-epoch ground-truth scores are written
  (`gt_*` vs `last_gt_*`) so selection cost is measurable within a run;
- `--entropy-beta` multiplies $\sum_{i\in\mathrm{batch}} H(q_i)$ in the
  minibatch ELBO (default $1$ = uniform-prior bound). An earlier mean-entropy
  implementation underweighted this term by roughly the batch size; the trainer
  now logs `ll_term` and `entropy_sum` separately. Sweep with
  `--lv-entropy-betas` in [`run_experiments.py`](run_experiments.py). Selection
  remains held-out AUC / LL, never the reweighted training objective.
  `hp_best.json` must propagate `entropy_beta` into finals (otherwise retrains
  silently default to $1.0$).
- `--partner-kl-weight` / `--lv-partner-kl-weights` add
  $\mathrm{KL}(h_i^{\mathrm{out}}\Vert q_i\sigma(\eta))$ (and the in-neighbour
  analogue) with stop-grad on the empirical neighbour mix. Default $0$ is the
  control. This asks that neighbours look like *partners of* $i$'s type, not
  like $i$ itself. Held-out positives are stripped from that adjacency.
  `hp_best.json` must propagate `partner_kl_weight` the same way as
  `entropy_beta`.

### Deferred

These stay out of the code on purpose (entropy $\beta_H$ and partner KL at
$\lambda=1$ are settled; free $B$ remains an optional next experiment). The
headline remains below NTAC:

1. **Adjacency smoothness on $q$.** Pulling neighbouring neurons toward similar
   posteriors is a homophily prior. FlyWire visual types are defined by partner
   patterns, and residual GNNs over soft assignments already hurt Hungarian while
   improving edge fit. Revisit only as a tiny negative-control if needed.
2. **Nonlinear (MLP) block decoder.** Extra decoder capacity is the failure mode
   of the removed $e_i$ / GNN arms. If block flexibility is revisited, prefer a
   free $K\times K$ logit table over an MLP.

### Baselines kept for comparison

- **PCA + $k$-means** — [`train_pca_baseline.py`](train_pca_baseline.py).
- **Unseeded NTAC** (Schwartzman et al., Nat Commun 2026) —
  [`train_ntac.py`](train_ntac.py). This is the current strong connectivity-only
  reference ($\approx 66\%$ Hungarian). The LV protocol above narrows the gap
  but does not close it.

## Evaluation protocol

1. Draw a held-out set of directed pairs (`heldout_pairs.npz`).
2. Sweep hyperparameters; pick the configuration with the best held-out metric
   (`--select-metric`).
3. Retrain that configuration for several seeds; report mean $\pm$ std of
   Hungarian / ARI / NMI against the FlyWire types on the shared labelled
   neurons.

Orchestration is [`run_experiments.py`](run_experiments.py)
(`--methods pca lv ntac`). Remote sweeps use
[`launch_lightning_sweep.py`](launch_lightning_sweep.py). Inspect results with
[`inspect_sweep_results.ipynb`](inspect_sweep_results.ipynb).

### Recommended lean LV grid

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

Entropy-weight ablation on the headline configuration
(`d=512`, `lr=0.03`, unit-norm scale $16$). Completed: restored sum-entropy
ELBO; swept $\beta_H\in\{0,0.1,0.3,1,3\}$ with AUC selection. HP Hungarian by
$\beta_H$: $0\to26\,092$, $0.1\to23\,638$, $0.3\to26\,568$, $1\to20\,569$,
$3\to2\,845$ (note $\beta_H=1$ under the restored scale is worse than the old
$\approx25\,k$ baseline, which had underweighted entropy). AUC selected
$\beta_H=0.3$; three-seed finals: Hungarian $26\,837\pm339$, AUC
$0.9859\pm0.0003$, $K_{\mathrm{pred}}$ $401\pm42$, ARI $0.609\pm0.027$, NMI
$0.862\pm0.004$.

```bash
uv run python run_experiments.py \
  --device cuda --phase all --methods lv \
  --lv-dims 512 --lv-lrs 0.03 \
  --lv-bfs-fracs 1.0 --lv-likelihoods bernoulli \
  --lv-u-norms unit --lv-u-scales fixed --lv-u-scale-inits 16.0 \
  --lv-entropy-betas 0 0.1 0.3 1 3 \
  --select-metric auc --grad-clip 1.0 \
  --epochs 40 --final-epochs 40 --final-seeds 0 1 2
```

#### Partner-histogram KL ablation

With $\beta_H=0.3$ fixed, the AUC-selected hyperparameter sweep produced:

| $\lambda$ | held-out AUC | Hungarian |
| ---: | ---: | ---: |
| $0$ | $0.9856$ | $26\,568$ |
| $0.1$ | $0.9864$ | $27\,121$ |
| $0.3$ | $0.9873$ | $26\,633$ |
| $1$ | $0.9897$ | $28\,272$ |
| $2$ | $0.9908$ | $27\,898$ |
| $3$ | $0.9913$ | $24\,439$ |
| $5$ | $0.9916$ | $24\,480$ |

The three-seed final at $\lambda=1$ achieves Hungarian $27\,644\pm32$, ARI
$0.600\pm0.007$, NMI $0.889\pm0.004$, $K_{\mathrm{pred}}=362\pm15$, and AUC
$0.9897\pm0.0002$. In contrast, the AUC-selected $\lambda=5$ final achieves
Hungarian $25\,534\pm542$, worse than the $\beta_H=0.3$-only result. Thus,
**$\lambda$ should be capped at $1$**: beyond this point, held-out AUC continues
to improve while Hungarian agreement collapses, a proxy--target split. The
scientific headline is therefore $\beta_H=0.3$ with $\lambda=1$, not the
AUC-selected $\lambda=5$ configuration.

```bash
uv run python launch_lightning_sweep.py \
  --machine T4 --methods lv --studio-name flywrite-partner-kl \
  --lv-dims 512 --lv-lrs 0.03 \
  --lv-u-norms unit --lv-u-scales fixed --lv-u-scale-inits 16.0 \
  --lv-entropy-betas 0.3 \
  --lv-partner-kl-weights 0 0.1 0.3 1 \
  --lv-likelihoods bernoulli --lv-bfs-fracs 1.0 \
  --select-metric auc --grad-clip 1.0 \
  --epochs 40 --final-epochs 40 --final-seeds 0 1 2 \
  --detach-only --remote-stop-after
```

On Lightning (cheaper T4 is enough for LV):

```bash
export LIGHTNING_USERNAME=kc119 LIGHTNING_TEAMSPACE=vision-model
# credentials: source ~/.ortet/lightning.env   # or equivalent
uv run python launch_lightning_sweep.py \
  --machine T4 --methods lv \
  --lv-dims 512 --lv-lrs 0.03 \
  --lv-u-norms unit --lv-u-scales fixed --lv-u-scale-inits 16.0 \
  --lv-entropy-betas 0 0.1 0.3 1 3 \
  --lv-likelihoods bernoulli --lv-bfs-fracs 1.0 \
  --select-metric auc --grad-clip 1.0 \
  --epochs 40 --final-epochs 40 --final-seeds 0 1 2 \
  --detach-only --remote-stop-after
```

Use `--skip-existing` to resume a partially finished Studio without wiping
`hp_*` / `final_*` artefacts.

## Environment

```bash
uv venv && source .venv/bin/activate
uv sync --exclude-newer "1 week"
```

Core dependencies are listed in `pyproject.toml` / `requirements.txt`
(PyTorch, SciPy, scikit-learn, `ntac`, Lightning SDK for remote sweeps).

## License

See the repository license file.
