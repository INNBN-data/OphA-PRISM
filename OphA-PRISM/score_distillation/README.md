# Developmental integer scores

`ophaprism.scores.integers` implements the supplied VISION (0–31) and CURE (0–33) score architectures. Inputs, bins, and weights are specified in the source code. Missing inputs raise an error; evaluation with missing predictors requires a locked training-derived imputation procedure.

A score total $S$ becomes a probability only after a validated mapping has been supplied:

$$
P(Y=1\mid S)=\operatorname{expit}(a+bS).
$$

`probabilities` requires both $a$ and $b$. The integer architecture is distinct from its six-variable probability surrogate. Derivation of the bins, weights, mapping, and risk strata requires evaluation within training partitions and subsequent external validation. This release provides score calculation; it does not reproduce the historical distillation process.
