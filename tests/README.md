### `ensemble_entropy_estimation.py`

- The first term of the BALD objective is the "entropy of average" of the 'ensemble'
- In our case, we have $I$ samples of the parameter $\theta$ from our MCMC estimation of the model posterior $p(\theta | \mathcal{D})$
- For a testpoint $x^\star$, each $\theta^{i}$ wil yield a new distribution $p(y^\star|\theta^{(i)}, x^\star)$ over the possible measurement at the testpoint
- The average of these distributions is $q(y^\star) = \frac{1}{I}\sum_{i=1}^{I}p(y^\star|\theta^{(i)}, x^\star)$
- We therefore need to estimate $H[q(y)]$, which we do by the following MCMC scheme:

1. For $i=1,...,I$:
    1. $y^{(i,k)} \sim p(y^\star|\theta^{(i)}, x^\star)$ for $k=1,...,K_i$
    2. $A_i = \sum_{i=1}^{K_i}\log\left(\frac{1}{I}\sum_{j=1}^{I} p(y^{(i,k)}|\theta^{(j)}, x^\star)\right)$
2. $H[q(y)]\approx\frac{1}{IK_i}A_i$

- We therefore need a tensor of shape `[I,I,K_i]` to compute this average for each testpoint!
- We will sweep values of I and K_i and run a couple of fidelity tests on the distribution
    - Namely, the distribution of $y^{(i,k)}$ against a test $q(y)$
- Remember that I also needs to be high enough to offer a good posterior estimation

