# Data Assimilation

# Introduction
Data Assimilation (DA) is the art of combining information from different sources in an optimal way. In climate and NWP, these sources are a first guess and observations with the aim of improving an estimate of the state of a system.
DA has two main objectives.

1) Produce an initial state from which we can start our forecast.
2) Quantifying the uncertainty of the initial state.

From the initial state of the atmosphere the model integrates a set of prognostic equations forward in time. This is prediction.

### What is the optimal/best analysis in operational NWP?
- The best analysis is the one that leads to the best forecast, not necessarily the analysis closest to the true state.
- Computational efficient: Depending on operational setup you should make your analysis in 30-60 minutes. Even your smartphones can compute a reasonable weather forecast faster than the weather evolves.
- Minimises error/maximises probability.

### Jargon in DA

Our prior information is a First Guess ($\mathbf{x_g}$) and some Observations ($\mathbf{y}$). $\mathbf{x_g}$ is the prior information and can for example be:

- A random guess (bad choice!)
- A climatological field (better, but room for improvement)
- The analysis from a previous cycle
∗ A short term forecast from the previous cycle $\mathbf{x_b}$
- Most often $\mathbf{x_g}=\mathbf{x_b}$ in operational setups.

We shall refer to a short term forecast as the background field $\mathbf{x_b}$, and denote our analysis as $\mathbf{x_a}$

### Observation coverage/density
The DMI Harmonie model, "NEA", has about $10^9$ prognostic variables that we need to assign an initial value to produce a forecast.
We have far less observations ($\approx 10^4$ to $10^6$) giving rise to an insufficient data coverage. Furthermore, observations are:

- Not evenly distributed in space and time
- Not taken exactly at the grid points.
- Not perfect - they contain errors. Quality control is needed.

An observation operator is needed to interpolate and convert to state variables if needed.

### A refresher on statistics

Suppose we have the noon-day pressure $p_i$ and temperature $T_i$ at Copenhagen, every day for a year. Let $n=365$.
The mean pressure, $\overline{p}$, is defined to be

$$
\overline{p}=\mathbb{E}(p)=\frac{1}{n}\sum_{i=1}^{n}p_i
$$

and similarly for the mean temperature $\overline{T}$.

The variance of pressure, $\sigma_p^2$, is defined as

$$
\sigma_p^2=\mathbb{E}((p-\overline{p})^2)=\frac{1}{n}\sum_{i=1}^{n}(p_i-\overline{p})^2
$$

and similarly for variance of the temperature $\sigma_T^2$.

The standard deviations, $\sigma_p$ and $\sigma_T$ are the square roots of the variances. They measure the root mean square deviation from the mean.