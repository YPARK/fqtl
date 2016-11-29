# Factored QTL analysis

## Probabilistic model

### Automatic confounder correction with individual-level data

In order to obtain high-quality z-score matrix we need to make sure no
significant effect of confounding variables leak into the z-scores.

1. Take phenotype matrix `Y` (n by m) and genotype matrix `X` (n by p).

2. Compute each column's effect size vector.  `Y[:,t] ~ X * theta` where we can use a different class of models.

3. Compute marginal: `E[Z] = X' * X * E[theta]` and `V[Z] = (X.*X)' * (X.*X) * V[theta]`

### Factored QTL estimation

