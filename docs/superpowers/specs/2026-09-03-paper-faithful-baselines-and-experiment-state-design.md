# Paper-faithful baselines and experiment-state design

## Goal

Make the BERT + linear-head experiments scientifically runnable by correcting
PraFFL and FedFACT-In against their papers and author implementations, making
three repeats/resume exact at communication-round boundaries, and replacing the
legacy quantity-skew split with deterministic label-conditioned Dirichlet
partitions shared by all algorithms.

## Sources of truth

- PraFFL paper: https://arxiv.org/abs/2404.08973
- PraFFL author code: https://github.com/rG223/PraFFL
- FedFACT paper: https://arxiv.org/abs/2506.03777
- FedFACT author code: https://github.com/liizhang/FedFACT
- NIID-Bench label-Dirichlet reference: https://github.com/Xtra-Computing/NIID-Bench/blob/main/partition.py

The papers define the algorithms. Author code resolves implementation details,
but obvious bugs in upstream code are not copied when they conflict with the
paper (for example, an off-by-one phase boundary or incorrect first-client
aggregation weight).

## Delivery structure

The work is split into three reviewable pull requests from the user's fork:

1. deterministic partitions, three repeats, and exact resume;
2. paper-faithful PraFFL adaptation;
3. paper-faithful FedFACT-In adaptation.

Each PR is test-first and independently reviewable. A combined integration
worktree is used for the final Ronnie BERT smoke test.

## 1. PraFFL adaptation

### Model boundary

For `BERTCLASSIFIER`, the communicated model is the BERT representation
encoder. Each client owns a private hypernetwork. Given a two-dimensional
preference vector, that hypernetwork generates the weight and bias of the
existing linear classification head. Generated parameters are consumed through
a functional linear operation, never copied with `.data`, so gradients reach
the hypernetwork.

### Per-round optimization

Each selected client performs two explicit phases:

1. `tau_c` communicated-model steps: use preference `(0.5, 0.5)`, freeze the
   private hypernetwork-generated head, and minimize cross entropy while
   updating the communicated encoder.
2. `tau_p` personalized steps: freeze and detach the encoder feature, draw
   preferences from `Dirichlet([1, 1])`, and update only the client's persistent
   hypernetwork using the differentiable DP surrogate and the inverse-weighted
   smooth Tchebycheff objective from the paper.

Only communicated encoder parameters are aggregated. Client hypernetworks are
private, persist across rounds, and are never averaged.

### Evaluation

The existing ACC/DEO/SPD table remains available at an explicitly named report
preference, default `(0.5, 0.5)`. In addition, PraFFL sweeps a deterministic
preference grid and records local/global Pareto points and hypervolume, because
that is the method's primary paper metric. Encoder features are computed once
per batch and reused across all preference heads.

## 2. FedFACT-In adaptation

The existing file is treated as an attempted FedFACT-In implementation; it is
not silently changed to FedFACT-Post.

Each client owns a persistent personal BERT classifier `phi_k`. In each round a
unified copy `theta_k` starts from the server model. Both are optimized with the
paper's client- and sensitive-group-conditioned calibrated cost-sensitive loss.
The prediction used for dual updates and evaluation is

`w_k * softmax(theta_k(x)) + (1 - w_k) * softmax(phi_k(x))`.

The ensemble weight uses the paper's exponentiated-loss update. Global and local
fairness constraints use separate non-negative positive/negative dual variables,
bounded projection, and explicit `global_constraint` and `local_constraint`
parameters. The global dual is updated from aggregated global disparity;
client-local duals are updated from each client's local disparity. Only
`theta_k` is uploaded and aggregated.

FedFACT evaluation must consume the personal models and ensemble weights. The
generic global-model-only evaluator is not used to label results as FedFACT.
Required support is checked before training: DP requires every client to contain
each sensitive group, while EO requires every `(label, sensitive-group)` cell.
Missing support raises a diagnostic configuration error; it is not smoothed or
silently treated as fair.

## 3. Three repeats and resume

- Serial and parallel execution share one repeat runner.
- Repeat `r` uses `base_seed + 1000 * r`; seed setup happens before model,
  sampler, or loader construction.
- Resume is opt-in. Without `-resume`, stale checkpoints are ignored.
- Every checkpoint records a canonical experiment-config hash, partition
  fingerprint, repeat index, round, Python/NumPy/CPU/CUDA RNG, AMP scaler, total
  runtime counters, and algorithm-specific state.
- A mismatch fails closed instead of partially restoring an incompatible run.
- A repeat is complete only after final evaluation is atomically written.
- Completed repeat metrics are loaded back into the three-repeat aggregate.
- A crash after the last training round but before final evaluation resumes at
  evaluation rather than incorrectly skipping the repeat.

For FedFACT's large personal states, only the latest resumable state for the
active repeat is retained. Completed matrix runs keep metrics/config and the
configured final artifact policy rather than multiplying all personal BERT
states across every configuration.

## 4. Label-conditioned Dirichlet

`Dirichlet01`, `Dirichlet05`, and `Dirichlet1` become versioned label-skew
partitions. For each class, a local NumPy generator samples client proportions;
the same class/client profile is used for train and client-level test allocation.
The implementation validates full coverage, uniqueness, bounds, and minimum
client size.

Sampling has a finite retry limit. If a valid allocation is not found, a
deterministic minimum-move repair fills undersized clients from donors while
preserving labels as much as possible. Repair count and resulting label/protected
statistics are stored in metadata; repaired output is never represented as an
unmodified pure draw.

Partition cache keys exclude algorithm/model and include schema version,
dataset/sample-order fingerprint, ordered-label fingerprint, alpha, number of
clients, repeat partition seed, minimum size, and repair policy. Thus the same
repeat is paired across algorithms, while changed data or split semantics cannot
reuse legacy indices. Legacy `split_indices.json` files remain readable only for
the legacy quantity-skew name and are never promoted to label-Dirichlet.

## 5. Resource constraints on Ronnie

- One experiment process and one active client training job use the single GPU.
- Repeats run serially on Ronnie.
- Only the active client's training models and optimizer states reside on GPU.
- PraFFL private hypernetworks and inactive FedFACT personal models reside on CPU.
- Checkpoint retention defaults to one latest resumable checkpoint for stateful
  algorithms.
- Smoke tests record peak CUDA allocation, host RSS, and checkpoint bytes before
  a full run is allowed.

## 6. Acceptance criteria

1. Unit tests first reproduce every identified failure.
2. PraFFL gradients reach both the communicated encoder and the active private
   hypernetwork in their respective phases; private hypernetworks are not equal
   merely because the server aggregated them.
3. FedFACT hand-computed cost, dual, probability-ensemble, and weight-update
   fixtures match paper/author-code behavior.
4. A continuous two-round run and a one-round-plus-resume run produce matching
   model, algorithm state, selected clients, and final metrics within declared
   numerical tolerance.
5. Three repeats produce three deterministic, distinct seeds and aggregate all
   three results even after mixed completed/resumed execution.
6. Same partition spec/repeat across two algorithms yields the same fingerprint
   and indices; changed data/order/alpha/seed yields a cache miss.
7. The former alpha=0.1, 40-client hang terminates deterministically.
8. Existing unit tests pass, followed by CPU toy tests and Ronnie AMP on/off BERT
   smoke tests.

## Non-goals

- Reproducing the papers' original tabular datasets or exact hyperparameters.
- Replacing BERT + linear head with the papers' MLP architectures.
- Changing FedAvg, LoGoFair, monitoring, or unrelated algorithms in these PRs.
- Treating old quantity-skew experiment results as comparable to new label-skew
  results.
