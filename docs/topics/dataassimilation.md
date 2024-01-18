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

<figure markdown>
  ![Image title](/images/single_obs.png){ width="400" }
  <figcaption>Increment of assimilating a single pressure observation</figcaption>
</figure>

In the figure the difference (increment) between to analysis states is shown. A single additional pressure observation is assimilated in the second analysis. The increment is largest close to the observation and decreases with distance. The spread of the increment is determined by $\mathbf{P}_b$.

### Solving for the gradient of the cost function

The minimum of the cost function is obtained for $\mathbf{x}=\mathbf{x}_a$, e.g. the solution of

$$
\nabla J(\mathbf{x}_a)=0
$$

So we wish to solve for the gradient of the cost function. To simplify the problem we linearise the observation operator $H$ as

$$
H(\mathbf{x})\approx \nabla H(\mathbf{x}_b)\cdot\delta \mathbf{x}+H(\mathbf{x}_b)
$$

and assume that the analysis is close to the truth so that we can write

$$
x=\mathbf{x}_b+(\mathbf{x}-\mathbf{x}_b)
$$

assuming $\mathbf{x}-\mathbf{x}_b$ is small, $\mathbf{y}-H(\mathbf{x})$ can be written as

$$
\begin{align*}
[\mathbf{y}-H(\mathbf{x})]=\mathbf{y}-H(\mathbf{x}_b+(\mathbf{x}-\mathbf{x}_b))\\
=\mathbf{y}-H(\mathbf{x})-\mathbf{H}(\mathbf{x}-\mathbf{x}_b)
\end{align*}
$$

This is an advantage as $H(\mathbf{x}_b)$ and $\mathbf{x}$ is known a priori. The cost function can now be written as

$$
\begin{align*}
J(\mathbf{x})=&\frac{1}{2}[(\mathbf{x}-\mathbf{x}_b)^T\mathbf{P}_b^{-1}(\mathbf{x}-\mathbf{x}_b)+[(\mathbf{y}-H(\mathbf{x}_b)-\mathbf{H}(\mathbf{x}-\mathbf{x}_b)]^T\mathbf{R}^{-1} \\
&[(\mathbf{y}-H(\mathbf{x}_b)-\mathbf{H}(\mathbf{x}-\mathbf{x}_b)])]]
\end{align*}
$$

If we then assume that $\mathbf{R}$ is symmetric so that $\mathbf{HR}^{-1}=\mathbf{R}^{-1}\mathbf{H}$ and expanding the brackets we get

$$
\begin{align*}
J(\mathbf{x})=&\frac{1}{2}[(\mathbf{x}-\mathbf{x}_b)^T\mathbf{P}_b^{-1}(\mathbf{x}-\mathbf{x}_b)+(\mathbf{x}-\mathbf{x}_b)^T\mathbf{H}^T\mathbf{R}^{-1}\mathbf{H}(\mathbf{x}-\mathbf{x}_b) \\
&-(\mathbf{y}-H(\mathbf{x}_b))^T\mathbf{R}^{-1}\mathbf{H}(\mathbf{x}-\mathbf{x}_b)-(\mathbf{x}-\mathbf{x}_b)^T\mathbf{H}^T\mathbf{R}^{-1} \\
&(\mathbf{y}-H(\mathbf{x}_b))+(\mathbf{y}-H(\mathbf{x}_b))^T\mathbf{R}^{-1}(\mathbf{y}-H(\mathbf{x}_b))]
\end{align*}
$$

If we combine the first two terms we get

$$
\begin{align*}
2J(\mathbf{x})=&\ (\mathbf{x}-\mathbf{x}_b)^T[\mathbf{P}_b^{-1}+\mathbf{H}^T\mathbf{R}^{-1}\mathbf{H}](\mathbf{x}-\mathbf{x}_b) \\
&-(\mathbf{y}-H(\mathbf{x}_b))^T\mathbf{R}^{-1}\mathbf{H}(\mathbf{x}-\mathbf{x}_b) \\
&-(\mathbf{x}-\mathbf{x}_b)^T\mathbf{H}^T\mathbf{R}^{-1}(\mathbf{y}-H(\mathbf{x}_b)) \\
&+\text(Term\ independent\ on\ \mathbf{x}) \\
=&\ (\mathbf{x}-\mathbf{x}_b)^T[\mathbf{P}_b^{-1}+\mathbf{H}^T\mathbf{R}^{-1}\mathbf{H}](\mathbf{x}-\mathbf{x}_b) \\
&-2(\mathbf{y}-H(\mathbf{x}_b))^T\mathbf{R}^{-1}\mathbf{H}(\mathbf{x}-\mathbf{x}_b) \\
&+\text(Term\ independent\ on\ \mathbf{x})
\end{align*}
$$

Given a quadratic function $F(\mathbf{x})=\frac{1}{2}\mathbf{x}^T\mathbf{Ax}+\mathbf{d}^T\mathbf{x}+c$ the gradient is given by $\nabla F(\mathbf{x})=\mathbf{Ax}+\mathbf{d}$. Using this and setting $\nabla J(\mathbf{x})=0$ to ensure $J$ is a minimum (though it could be a maximum or a saddle point) we obtain an analytical expression for the analysis state vector $\mathbf{x}_a$.

$$
\begin{align*}
\mathbf{x}_a = \mathbf{x}_b+(\mathbf{P}_b^{-1}+\mathbf{H}^T\mathbf{R}^{-1}\mathbf{H})^{-1}\mathbf{H}^T\mathbf{R}^{-1}(\mathbf{y}-H(\mathbf{x}_b))
\end{align*}
$$

The is the analytical solution to {==3DVar==}. In practice it is solved by an iterative method such as the steepest descent method, as we saw in the scalar example. Unfortunately for us, it is impossible to invert such huge matrices as $\mathbf{P}_b$ so we need to find approximations. Also {==3DVar==} assumes all observations to be taken at the time of the analysis. This is not the case in reality. $\mathbf{P}_b$ is also assumed to be constant in time, which is not the case in reality (not allowed to evolve dynamically).


<figure markdown>
  ![Image title](/images/3dvar.png){ width="400" }
  <figcaption>Schematic figure of the 3DVar algorithm</figcaption>
</figure>

## Kalman-based Methods
Optimal Interpolation, Kalman Filter and Kalman Ensemble filters are methods that are widely used in operational centers and they are all based on the Kalman equations, which we shall derive to broaden our understanding of the methods.

An analysis is found by using a least square approach in the sense that we find an ’optimal’ analysis by minimising the errors.

We write our analysis $\mathbf{x}_a$ as a linear combination of the background, $\mathbf{x}_b$ and some observations $\mathbf{y}$ as

$$
\mathbf{x}_a=\mathbf{Lx}_b+\mathbf{Wy}
$$

where $\mathbf{L}$ and $\mathbf{W}$ are weights that we need to find.

One can derive the least square solution from this equation and at the same time get rid of one of the weights. This is a statistical approach trying to minimise the errors of the analysis.

The errors of $\mathbf{x}_a$ and $\mathbf{x}_b$ can be written as

$$
\begin{align*}
\mathbf{e}_a=&\ \mathbf{x}_a-\mathbf{x}_t \\
\mathbf{e}_b=&\ \mathbf{x}_b-\mathbf{x}_t
\end{align*}
$$

where $\mathbf{x}_t$ is the true (unknown) state. A linear observation process can be defined as

$$
\mathbf{y}=\mathbf{Hx}_t+\mathbf{b}_0
$$

where $\mathbf{H}$ is a matrix representing a linear transformation between the true variables into the observed ones (also called a forward operator) and $\mathbf{b}_0$ is the observational error.

Assume that the observation error have zero mean and covariance $\mathbf{R}$,

$$
\begin{align*}
\mathbb{E}(\mathbf{b}_0) =& 0 \\
\mathbb{E}(\mathbf{b}_0\mathbf{b}_0^T) =& \mathbf{R}_k\delta_{kk'}
\end{align*}
$$

where $\delta_{kk'}$ is the dirac-delta function.

Also assume that the observations errors and model errors are uncorrelated

$$
\mathbb{E}(\mathbf{b}_t\mathbf{b}_0^T)=0
$$

Substitung the errors of the analysis and background into our analysis equation and subtracting $\mathbf{x}_t$ we get

$$
\mathbf{e}_a=\underbrace{\mathbf{Le}_b}_{\text{Background Error}}+\underbrace{\mathbf{Wb}_0}_{\text{Observational Error}}+\underbrace{(\mathbf{L}+\mathbf{WH}-\mathbf{I})\mathbf{x}_t}_{\text{Bias}}
$$

Assuming that the forecast error is unbiased ($\mathbb{E}(\mathbf{e}_b)=\mathbb{E}(\mathbf{x}_b-\mathbf{x}_t)=0$), the condition $(\mathbf{L}+\mathbf{WH}-\mathbf{I})\mathbb{E}(\mathbf{x}_t)=0$ must be met. In general $\mathbb{E}(\mathbf{x}_t)\neq 0$ so to obtain an unbiased analysis we can write the first weight in terms of the second as

$$
\mathbf{L}=\mathbf{I}-\mathbf{WH}
$$

Substituting this into the analysis equation we get the Kalman analysis equation

$$
\begin{align*}
\mathbf{x}_a=&\mathbf{x}_b+\mathbf{W}\underbrace{(\mathbf{y}-\mathbf{Hx}_b)}_{\text{Innovation}} \\
=&\mathbf{x}_b+\mathbf{Wd}
\end{align*}
$$

### Dimension

$n=\text{Total number of grid points}\times\text{number of model variables}$

$\mathbf{x}$: State vector of size $n$

$\mathbf{W}$: Weight matrix of size $p\times n$ where $p$ is the number of observations

$\mathbf{y}$: Observation vector of size $p$

$\mathbf{H}$: Matrix of size $n\times p$

### Derivation of the weight

At this point $\mathbf{W}$ is still unknown to us. $\mathbf{W}$ chosen such that the variances are minimised. Consider the error covariance of the analysis,

$$
\begin{align*}
	\mathbf{P}_a=\text{cov}[\mathbf{x}_t-\mathbf{x}_a]=
	\left[ {\begin{array}{cccc}
	\sigma_{1,1}^2 & \sigma_{1,2}^2 & \dots & \sigma_{1,n}^2 \\
	\sigma_{2,1}^2 & \sigma_{2,2}^2 & \dots & \sigma_{2,n}^2 \\
	\vdots         & \vdots         & \ddots & \vdots \\
	\sigma_{m,1}^2 & \sigma_{m,2}^2 & \dots & \sigma_{m,n}^2
	\end{array} } \right]
\end{align*}
$$

Note that the variances are the trace of the error covariance matrix. This can be expanded by using the Kalman analysis equation and $\mathbf{y}=\mathbf{Hx}_t+\mathbf{b}_0$ to get

$$
\begin{align*}
	\mathbf{P}_a = \text{cov}[(\mathbf{I}-\mathbf{WH})(\mathbf{x}_t-\mathbf{x}_b)-\mathbf{Wb}_0].
\end{align*}
$$

This can be simplified by using the covariance matrix identity, $\text{cov}(\mathbf{AB})=\mathbf{A}\text{cov}(\mathbf{B})\mathbf{A}^T$:

$$
\begin{align*}
	\mathbf{P}_a = (\mathbf{I}-\mathbf{WH})\mathbf{P}_b(\mathbf{I}-\mathbf{WH})^T+\mathbf{WRW}^T,
\end{align*}
$$

where $\mathbf{P}_b=\text{cov}(\mathbf{x}_t-\mathbf{x}_b)$ and $\mathbf{R}=\text{cov}(\mathbf{b}_0)$.

We expand by using that $\mathbf{I}=\mathbf{I}^T$ and defining $\mathbf{S}=\mathbf{HP}_b\mathbf{H}^T+\mathbf{R}$ to get

$$
\begin{align*}
	\mathbf{P}_a = \mathbf{P}_b - \mathbf{W}^T\mathbf{H}^T\mathbf{P}_b-\mathbf{WHP}_b+\mathbf{WSW}^T.
\end{align*}
$$

Recall that we want to minimise the trace of the error covariance matrix (to minimise the variances). We take the derivative of the trace of $\mathbf{P}_a$ with respect to $\mathbf{W}$ and setting it equal to 0 to find the minimum (hint: Use the matrix identity $\nabla_A\text{Tr}(\mathbf{AB})=\mathbf{B}^T$).

$$
\begin{align*}
	\frac{\partial\text{Tr}(\mathbf{P}_a)}{\partial\mathbf{W}} = -2(\mathbf{HP}_b)^T + 2\mathbf{WS} \equiv 0.
\end{align*}
$$

Using that $\mathbf{P}_b$ is symetric ($\mathbf{P}_b=\mathbf{P}_b^T$) and solving for $\mathbf{W}$ yields the optimal weight

$$
\begin{align*}
	\mathbf{W}=\mathbf{H}^T\mathbf{P}_b\mathbf{S}^{-1} = \frac{\mathbf{H}^T\mathbf{P}_b}{\mathbf{HP}_b\mathbf{H}^T+\mathbf{R}}
\end{align*}
$$

This is called the Kalman gain and is the optimal weight that minimises the variances of the analysis.

### Understanding the Kalman gain
It is important to get an intuitive understanding of the Kalman gain. The Kalman gain is a measure of how much we trust the observations. If the observation error is large, the Kalman gain will be small and vice versa. If the background error is large, the Kalman gain will be small and vice versa.

Recall that $\mathbf{R}$ is the observational error covariance matrix and $\mathbf{P}_b$ is the background error covariance matrix.

If the {==observational error==} is much larger than the {==model error==} the Kalman gain will go towards 0, making the innovation term in equation small, such that observations are given a low weight.

On the other hand, if the {==model error==} is much larger than the {==observational error==} the weight will go towards 1, given the innovation term a high weight in the analysis equation.

One advantage of the Kalman system is that we get the error of the analysis directly by computing $\mathbf{P}_a$. This is not the case for the variational methods. However, we can now simplify the equation for $\mathbf{P}_a$ by multiplying the Kalman gain with $\mathbf{W}^T\mathbf{S}$ to get

$$
\mathbf{W}^T\mathbf{SW} = \mathbf{P}_b\mathbf{H}^T\mathbf{W}^T
$$

Substituting this into the equation for $\mathbf{P}_a$ we get

$$
\mathbf{P}_a=(\mathbf{I}-\mathbf{WH})\mathbf{P}_b
$$

The Kalman filter and OI methods are very similar with some important differences though. In the Kalman Filter, $\mathbf{P}_b$ is dynamic, hence updated with each analysis. In Optimal Interpolation (OI) $\mathbf{P}_b$ is static, hence constant in time. Due to the dynamic $\mathbf{P}_b$ in the Kalman Filter it is for example used partly in auto-piloting in airplanes and self-driving cars.

### The Kalman Filter Algorithm

The Kalman Filter algorithm is a recursive algorithm which has a prediction step and an update step.

{==Prediction step==}

$$
\begin{align*}
\mathbf{x}_f=&\mathbf{Mx}_a \\
\mathbf{P}_f=&\mathbf{MP}_a\mathbf{M}^T+\mathbf{Q}
\end{align*}
$$

{==Update step==}

$$
\begin{align*}
\mathbf{K}=& \mathbf{P}_f\mathbf{H}^T(\mathbf{HP}_f\mathbf{H}^T+\mathbf{R})^{-1} \\
\mathbf{x}_a=&\mathbf{x}_f+\mathbf{K}(\mathbf{y}-\mathbf{Hx}_f) \\
\mathbf{P}_a=&(\mathbf{I}-\mathbf{KH})\mathbf{P}_f
\end{align*}
$$

Here $\mathbf{K}=\mathbf{W}$ is used for consistency with the literature. $\mathbf{Q}$ is the forecast error covariance. $\mathbf{M}$ is the linear tangent model and $\mathbf{M}^T$ is its adjoint. $\mathbf{M}$ is the operator that forwards the model in time from the analysis.

## Optimal Interpolation (OI) equations

Analysis Equation:

$$
\begin{align*}
	\mathbf{x}_a=\mathbf{x}_b+\mathbf{W}(\mathbf{y}-\mathbf{Hx}_b)=\mathbf{x}_b+\mathbf{Wd}
\end{align*}
$$

Optimal Weight:

$$
\begin{align*}
	\mathbf{W}=\mathbf{H}^T\mathbf{P}_b\mathbf{S}^{-1} = \frac{\mathbf{H}^T\mathbf{P}_b}{\mathbf{HP}_b\mathbf{H}^T+\mathbf{R}}
\end{align*}
$$

Analysis Error Covariance:

$$
\begin{align*}
	\mathbf{P}_a = (\mathbf{I}-\mathbf{WH})\mathbf{P}_b
\end{align*}
$$

$\mathbf{P}_b$ is static and is usually computed by running a model over a long period (weeks to months) and looking at the error statistics. Therefore $\mathbf{P}_b$ must be updated for every change in model configuration (dynamics, physics, domain).

## Characteristic overview of DA methods

|       | Method      |        | Observations |          | Covariance |         |
|-------|-------------|:------:|--------------|:--------:|------------|:-------:|
|       | Variational | Kalman | Sequential   | Smoother | Static     | Dynamic |
| 3DVar |      x      |        |       x      |          |      x     |         |
| 4DVar |      x      |        |              |     x    |     (x)    |    x    |
| OI    |             |    x   |       x      |          |      x     |         |
| KF    |             |    x   |       x      |          |            |    x    |
