# Gaussian Process for Time-To-Event Analysis

## Methodology

1. semiparametric Bayesian model
2. able to handle censoring and covariates
3. models the hazard function as the multiplication of a parametric baseline hazard and a nonparametric part
4. parametric part of our model allows
the inclusion of expert knowledge and provides interpretability
5. while the nonparametric part allow
us to handle covariates and to amend incorrect or incomplete prior knowledge
6. nonparametric
part is given by a non-negative function of a Gaussian process on $\mathbb{R}^+$

### Definition

$$
\lambda = f / S \\
$$
$
\begin{aligned}where \qquad
\lambda&=\text{harzard function}\\
f&=\text{density function}\\
S&=\text{survival function}
\end{aligned}
$

