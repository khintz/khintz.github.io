# Data Assimilation

## Introduction
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

### Framework of DA
As has been described we have information both from a short term forecast, $\mathbf{x_b}$, and some observations $\mathbf{y}$. We need to develop a framework for combining the two sources of information.

Luckily this framework already exist! → {==Bayes Theorem==}

$$
\text{pdf}(\mathbf{x}|\mathbf{y})=\frac{\text{pdf}(\mathbf{y}|\mathbf{x})\text{pdf}(\mathbf{x})}{\text{pdf}(\mathbf{y})}
$$

- $\text{pdf}(\mathbf{x}|\mathbf{y})$ is the posterior probability density function (pdf) of $\mathbf{x}$ given $\mathbf{y}$
- $\text{pdf}(\mathbf{y}|\mathbf{x})$ is the likelihood function, the $\text{pdf}$ of the observations given the state variables.
- $\text{pdf}(\mathbf{x})$ is the prior $\text{pdf}$ of the state variables coming from the background field (model)
- $\text{pdf}(\mathbf{y})$ is the evidence. It is used as a normalisation constant and often not computed explicitly so we will ignore this for now.


Combining information in the form of the prior and the likehood gives us a more narrow posterior $\text{pdf}$. Note that as long one knows the associated error of either the model or observations the posterior will always gain information.

<figure markdown>
  ![Image title](/images/prob_distribution.png){ width="600" }
  <figcaption>Probability Density Functions</figcaption>
</figure>

### Reality bites

Making no approximations - considering the full non-linear DA problem - we have to find the joint $\text{pdf}$ of all variables, that is the probabilities for all possible combinations of all of our variables.

Assume we only need 10 bins for each variable to generate a joint $\text{pdf}$ and assume we have a small model of only $10^6$ variables. Then we need to store $10^{1000000}$ numbers.

There are approximately $10^{80}$ atoms in the universe[^1].
So data assimilation is much larger than the universe!
We need $10^{52}$ solar system sized hard drives to store just a googol ($10^{100}$) bytes, but we only have about $10^{24}$ stars[^2]. Approximations and optimisations are indeed needed.

The key here is that we can’t determine the pdf’s in large dimensional systems.

[^1]: https://www.universetoday.com/36302/atoms-in-the-universe/
[^2]: https://www.quora.com/How-much-hard-drive-storage-would-you-need-in-your-computer-to- fully-type-out-the-number-Googolplexian

### Approximate solutions

Estimating the $\text{pdf}$’s in large dimensional systems, such as in NWP, is practically impossible!

Approximate solutions of Bayes theorem leads to data assimilation methods.

- {==Variational methods==}: Solves for the mode of the posterior pdf.
- {==Kalman-based methods==}: Solves for the mean and covariance of posterior pdf.
- {==Particle Filters==}: Finds a sample representaion of the posterior pdf.

{==Variational methods==} and {==Kalman-based methods==} both assume errors to be Gaussian. {==Particle Filters==}, however have no such assumptions.

## Variational Methods
The variational methods assume errors to be Gaussian. This is convenient as the pdf is completely determined by the mean and covariance. But it is also a strong constraint. Do the errors, in fact, follow a Gaussian?

In geophysics, we often deal with only the positive real axis, where we then often find a tail on the error distribution. For example precipitation, wind, salinity etc.

Let us see how the variational approach works with a scalar example of temperature observations.

We will feed our own ”toy data assimilation system” the observations, to obtain the most likely temperature given your observations.

### The cost function

We assume that your observations has been drawn from a Gaussian distribution.

$$
p(T_1|T)=\frac{1}{\sqrt{2\pi\sigma_1^2}}\exp\left(-\frac{(T_1-T)^2}{2\sigma_1^2}\right)
$$

and likewise for $T_2$. We can express the joint probability by multiplying the two probabilities together.

$$
p(T_1,T_2|T)=\frac{1}{2\pi\sigma_1\sigma_2}\exp\left(-\frac{(T_1-T)^2}{2\sigma_1^2}-\frac{(T_2-T)^2}{2\sigma_2^2}\right)
$$

This is the same as the likelihood for $T$ given $T_1$ and $T_2$, $L(T|T_1,T_2)$. To find the most likely $T$, which will be our analysis temperature $T_a$, we want to maximise the likelihood given $T_1$ and $T_2$.

$$
\text{max}[L(T|T_1,T_2)]=\text{max}\left[\frac{1}{2\pi\sigma_1\sigma_2}\exp\left(-\frac{(T_1-T)^2}{2\sigma_1^2}-\frac{(T_2-T)^2}{2\sigma_2^2}\right)\right]
$$

To make things easier, we take the logarithmic of this expression. Note that the logarithmic is a monotonic function, so maximising the logarithmic of the function is equivalent to maximising the function itself.

$$
\text{max}[\ln L(T|T_1,T_2)]=\text{max}\left[\text{const}-\frac{1}{2}\left(\frac{(T-T_1)^2}{\sigma_1^2}+\frac{(T-T_2)^2}{\sigma_2^2}\right)\right]
$$

Note that maximising the function is equivalent to minimising the last term on the right-hand-side. For our ”two-temperature-problem” this is defined as our cost function.

$$
J=\frac{1}{2}\left(\frac{(T-T_1)^2}{\sigma_1^2}+\frac{(T-T_2)^2}{\sigma_2^2}\right)
$$

Minimising $J$ corresponds to maximising the likelihood of $T$ given $T_1$ and $T_2$.

To minimise $J$ to find our analysis temperature using the variational approach, we will start by guessing some value of $T$ and explore space. Different algorithms exist for this such as "steepest-descent".

??? example

    ```python
    """
    'Toy-Tool' to play with a very simple scalar case of the cost function
    in variational data assimilation.

    Given two guesses of temperature, we try to find the most likely
    true temperature by minimizing a cost function.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Define variables
    T1 = 21.6  # First guess of temperature
    sigma1 = 1.8  # Standard deviation for T1

    T2 = 23.4  # Second guess of temperature
    sigma2 = 0.8  # Standard deviation for T2

    # Initial guess (mean of T1 and T2)
    T0 = (T1 + T2) / 2

    # Initialize variables
    T = T0  # Current temperature guess
    eps = 0.1  # Epsilon for perturbation
    J0 = 2000.  # Initial cost value

    # Lists to store iteration data
    iteration = []
    Ta = []
    costF = []

    # Perform 100 iterations
    for k in range(100):
        direction = np.random.randint(-1, 2) + eps
        size_direction = np.random.rand(1) * direction

        Tg = T + size_direction  # New temperature guess
        J = 1/2 * ((Tg - T1)**2 / sigma1**2 + (Tg - T2)**2 / sigma2**2)  # Cost function

        # Update if new cost is lower
        if J < J0 and k > 0:
            J0 = J
            T = Tg
            iteration.append(k)
            Ta.append(T)
            costF.append(J)
            print(T, J)

    # Plotting
    fig = plt.figure()
    plt.plot(Ta, iteration)
    plt.gca().invert_yaxis()
    plt.ylabel('Iteration')
    plt.xlabel('T [C]')
    plt.show()

    ```

### Probability distributions for the multivariate case

The principles we have just seen, are the same in the multivariate case.

The {==prior==}

$$
p(\mathbf{x})\propto\exp\left(-\frac{1}{2}(\mathbf{x}-\mathbf{x_b})^T\mathbf{P}^{-1}(\mathbf{x}-\mathbf{x_b})\right)
$$

The {==likelihood==}

$$
p(\mathbf{y}|\mathbf{x})\propto\exp\left(-\frac{1}{2}(\mathbf{y}-H(\mathbf{x}))^T\mathbf{R}^{-1}(\mathbf{y}-H(\mathbf{x}))\right)
$$

The {==posterior==}

$$
p(\mathbf{x}|\mathbf{y})\propto\exp\left(-\frac{1}{2}[(\mathbf{x}-\mathbf{x_b})^T\mathbf{P}_b^{-1}(\mathbf{x}-\mathbf{x_b})+(\mathbf{y}-H(\mathbf{x}))^T\mathbf{R}^{-1}(\mathbf{y}-H(\mathbf{x}))]\right)
$$

#### Notation

$\mathbf{x}$: State vector of size $n$

$\mathbf{x_b}$: Background state vector of size $n$

$\mathbf{P}_b$ or $\mathbf{B}$: Background error covariance matrix of size $n\times n$

$\mathbf{y}$: Observation vector of size $p$

$\mathbf{R}$: Observation error covariance matrix of size $p\times p$

$n$: Total number of grid points $\times$ number of model variables ($\approx 10^7$)

$p$: Total number of observations ($\approx 10^4$)

$H(\mathbf{x})$: Observation operator that maps model space to observation space. $H(x_i)$ is the models estimate on $y_i$. $H$ can be non-linear (eg. radiance measurements) or linear (eq. synop temperature measurements).

### Cost function multivariate case
Taking the term in the brackets of the {==posterior==} gives us the cost function that we want to minimise to maximise the posterior.

$$
J(\mathbf{x})=\frac{1}{2}[(\mathbf{x}-\mathbf{x_b})^T\mathbf{P}_b^{-1}(\mathbf{x}-\mathbf{x_b})+(\mathbf{y}-H(\mathbf{x}))^T\mathbf{R}^{-1}(\mathbf{y}-H(\mathbf{x}))]
$$

where $\mathbf{x}$ is the state vector with all variables. The $\mathbf{x}$ that minimises the cost function, $J$ is our analysis state vector $\mathbf{x}_a$. $\mathbf{P}_b$ is the background error covariance matrix. $\mathbf{P}_b$ is sometimes also denoted $\mathbf{B}$ in the literature. Here I use $\mathbf{P}_b$ for consistency with the Kalman Filter algorithm (to be introduced)

The cost function cannot be computed directly. Different assumptions leads to either {==3DVar==} or {==4DVar==}.

$\mathbf{P}_b$ is a huge matrix! ($\approx 10^7\times10^7$) - We can’t store this on a computer, so we are forced to simplify it. Furthermore, it is not given that we have all the information needed to determine all of its elements.

We dont know $\mathbf{P}_b$. In general $\mathbf{P}_b = \text{cov}[\mathbf{x}_t − \mathbf{x}_b]$, but we have no way to know $\mathbf{x}_t$, therefore a proxy is needed for $\mathbf{x}_t$.

As a proxy for $\mathbf{x}_t$ one can use ”observation-minus-background” statistics, by running the model for a long period and see how the error looks in average.

$\mathbf{P}_b$ is essential. Its role is to spread out information from observations. How should a pressure observation in Copenhagen affect variables in Oslo? Also, it ensures dynamically consistent increments in other model variables (How should a temperature increase affect the wind?)

??? tip

    Good To Know: $\mathbf{P}_b$ is often referred to as ”structure functions” in the literature.