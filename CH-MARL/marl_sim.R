% =============================================================================
% Simulation Study
% =============================================================================

\section{Simulation Study}
\label{sec:simulation}

We conduct a controlled simulation study to evaluate the proposed
\emph{Copula-Hierarchical Multi-Agent Reinforcement Learning} (CH-MARL)
framework against two established multi-agent reinforcement learning
baselines, independent proximal policy optimization (IPPO) and multi-agent
proximal policy optimization (MAPPO). The simulation is designed to examine
both task performance and the learning dynamics associated with centralized
coordination, hierarchical goal generation, and inter-agent dependence.

\subsection{Simulation Environment}
\label{subsec:simulation_environment}

The simulated environment consists of $N=8$ heterogeneous agents operating
in a bounded two-dimensional workspace,
\[
\mathcal{X}=[0,2]\times[0,2].
\]
Initial agent locations are generated randomly within the interior region
$[0.1,1.9]^2$ using a fixed random seed of 42 to ensure reproducibility.
The agents are assigned heterogeneous mobility characteristics by randomly
sampling one of three agent types: \emph{scout}, \emph{patrol}, and
\emph{heavy}. Their corresponding movement speeds and effective radii are
\[
\begin{array}{c|cc}
\text{Agent type} & \text{Speed} & \text{Radius}\\
\hline
\text{Scout} & 0.25 & 0.40\\
\text{Patrol} & 0.15 & 0.25\\
\text{Heavy} & 0.08 & 0.15
\end{array}
\]
respectively. This heterogeneity prevents the coordination problem from
reducing to a collection of identical agents with interchangeable dynamics.

The common navigation objective is a target located at
\[
\boldsymbol{g}=(1.8,1.8).
\]
The environment contains two fixed obstacles at
$(0.5,0.5)$ and $(0.8,1.2)$, together with a time-varying obstacle whose
location at step $t$ is
\[
\boldsymbol{o}_t =
\left(
1.0+0.3\sin(0.1t),
1.0+0.3\cos(0.1t)
\right).
\]
Thus, agents must coordinate their movements in the presence of both
stationary and dynamic spatial constraints.

At each step, an agent selects a two-dimensional continuous action
\[
\boldsymbol{a}_{i,t}\in[-1,1]^2.
\]
Given its agent-specific speed $v_i$, the unconstrained position update is
\[
\widetilde{\boldsymbol{x}}_{i,t+1}
=
\boldsymbol{x}_{i,t}
+
v_i\boldsymbol{a}_{i,t},
\]
followed by projection onto the workspace:
\[
\boldsymbol{x}_{i,t+1}
=
\Pi_{[0,2]^2}
\left(
\widetilde{\boldsymbol{x}}_{i,t+1}
\right).
\]

An agent is considered to have reached the target when its squared Euclidean
distance from the target satisfies
\[
\left\|
\boldsymbol{x}_{i,t+1}-\boldsymbol{g}
\right\|^2
\leq 0.09.
\]
Once an agent reaches the target, its completion status is persistent for the
remainder of the episode and the agent becomes inactive. The episode
terminates when all agents have reached the target or when the maximum
number of execution steps, $T_{\max}=100$, is reached.

The individual reward is defined as
\[
r_{i,t}=
\begin{cases}
-1, & \text{if agent $i$ collides with an obstacle},\\[3pt]
2, & \text{if agent $i$ reaches the target},\\[3pt]
-0.01+
0.05\left(1-\dfrac{d_{i,t}}{2}\right),
& \text{otherwise},
\end{cases}
\]
where
\[
d_{i,t}
=
\left\|
\boldsymbol{x}_{i,t+1}-\boldsymbol{g}
\right\|.
\]
The global reward used for training is the mean reward across agents,
\[
r_t=\frac{1}{N}\sum_{i=1}^{N}r_{i,t}.
\]
Consequently, the learning objective rewards collective progress while
retaining individual collision and target-completion effects.

\subsection{State Representation and Phase Embedding}
\label{subsec:state_representation}

The raw state of agent $i$ is its two-dimensional position
$\boldsymbol{x}_{i,t}=(x_{i,t}^{(1)},x_{i,t}^{(2)})$. To provide a richer
representation of spatial state, each coordinate is augmented using a
sinusoidal phase embedding. Specifically, for a scalar state component
$x$, the embedding is
\[
\phi(x)=
\left[
x,\,
\sin(\pi x),\,
\cos(\pi x)
\right].
\]
The resulting six-dimensional representation of agent $i$ is
\[
\boldsymbol{s}_{i,t}
=
\left[
x_{i,t}^{(1)},
x_{i,t}^{(2)},
\sin(\pi x_{i,t}^{(1)}),
\sin(\pi x_{i,t}^{(2)}),
\cos(\pi x_{i,t}^{(1)}),
\cos(\pi x_{i,t}^{(2)})
\right]^{\top}.
\]
The joint state is obtained by concatenating the embedded states of all
agents,
\[
\boldsymbol{s}_t
=
[
\boldsymbol{s}_{1,t}^{\top},
\ldots,
\boldsymbol{s}_{N,t}^{\top}
]^{\top}
\in\mathbb{R}^{6N}.
\]
For $N=8$, the joint state dimension is therefore 48.

\subsection{Compared MARL Algorithms}
\label{subsec:compared_algorithms}

Three algorithms are evaluated using the same simulated environment,
training horizon, action dimension, and optimization settings.

\paragraph{CH-MARL.}
The proposed CH-MARL architecture combines hierarchical goal generation,
stochastic agent policies, and a Gaussian copula representation of
inter-agent action dependence. A manager network maps the joint state to a
four-dimensional latent coordination goal,
\[
\boldsymbol{g}_t^{\,M}
=
f_M(\boldsymbol{s}_t)
\in[-1,1]^4.
\]
Each worker then receives its own six-dimensional phase-embedded state
together with the manager output:
\[
\boldsymbol{h}_{i,t}
=
[
\boldsymbol{s}_{i,t}^{\top},
(\boldsymbol{g}_t^{\,M})^{\top}
]^{\top}.
\]
The worker produces the parameters of a stochastic continuous-action policy,
\[
(\boldsymbol{\mu}_{i,t},
\boldsymbol{\ell}_{i,t})
=
f_i(\boldsymbol{h}_{i,t}),
\]
where $\boldsymbol{\ell}_{i,t}$ denotes the logarithm of the action standard
deviation.

Unlike independent action sampling, CH-MARL models dependence among the
$Nd=16$ action components through a Gaussian copula. A neural copula
network receives the joint state and produces an unconstrained scalar
$\rho_t^{\mathrm{raw}}$. This value is transformed to a valid equicorrelation
parameter according to
\[
\rho_t
=
\rho_{\min}
+
(\rho_{\max}-\rho_{\min})
\sigma(\rho_t^{\mathrm{raw}}),
\]
where
\[
\rho_{\min}
=
-\frac{1}{Nd-1}+\epsilon,
\qquad
\rho_{\max}=0.95,
\qquad
\epsilon=10^{-4},
\]
and $\sigma(\cdot)$ is the logistic function. The corresponding correlation
matrix is
\[
\mathbf{R}_t
=
(1-\rho_t)\mathbf{I}_{Nd}
+
\rho_t\mathbf{1}_{Nd}\mathbf{1}_{Nd}^{\top}.
\]
This parameterization guarantees a valid positive-definite equicorrelation
matrix throughout training.

\paragraph{IPPO.}
IPPO provides a decentralized baseline in which each agent has its own
stochastic worker policy and its own critic. Each critic receives only the
corresponding agent's phase-embedded state and action,
\[
(\boldsymbol{s}_{i,t},\boldsymbol{a}_{i,t}),
\]
and estimates an individual action-value function. The action components are
sampled independently, without a learned inter-agent dependence structure.
Thus, IPPO provides a baseline for decentralized policy learning without
hierarchical coordination or explicit copula dependence.

\paragraph{MAPPO.}
MAPPO uses decentralized stochastic worker policies together with a
centralized critic. The centralized critic receives the complete joint state
and joint action,
\[
(\boldsymbol{s}_t,\boldsymbol{a}_t),
\]
and therefore has access to global information during value estimation.
Nevertheless, the policies themselves do not contain the hierarchical
manager or the copula-based action-dependence mechanism used by CH-MARL.
MAPPO consequently provides a centralized-critic baseline against which the
additional contributions of hierarchical coordination and dependence
modeling can be examined.

\subsection{Neural Network Architecture}
\label{subsec:network_architecture}

All neural networks are implemented using Keras 3 and TensorFlow. The
manager network consists of two fully connected hidden layers with 32 units
each and ReLU activation, followed by a four-dimensional hyperbolic tangent
output layer. Each stochastic worker consists of two hidden layers with 32
and 16 ReLU units, respectively. Separate output layers produce the action
mean and log standard deviation.

The critic architecture consists of fully connected layers with 64 and
32 ReLU units followed by a scalar linear output. IPPO uses one such critic
for each agent, whereas CH-MARL and MAPPO use a centralized critic operating
on the joint state-action representation. The CH-MARL copula network consists
of 64- and 32-unit ReLU hidden layers followed by a scalar output representing
the unconstrained correlation parameter.

\subsection{Training Procedure}
\label{subsec:training_procedure}

Each model is trained for 25 episodes, with a maximum of 100 environment
steps per episode. The discount factor is
\[
\gamma=0.99.
\]
Actor and critic learning rates are set to
\[
\eta_{\mathrm{actor}}=5\times10^{-4},
\qquad
\eta_{\mathrm{critic}}=10^{-3},
\]
respectively, and the entropy coefficient is
\[
\beta=0.001.
\]
A replay buffer with capacity 50,000 transitions is used, with a minibatch
size of 32. Target critics are updated using soft target updates,
\[
\boldsymbol{\theta}^{\mathrm{target}}
\leftarrow
\tau\boldsymbol{\theta}
+
(1-\tau)\boldsymbol{\theta}^{\mathrm{target}},
\]
where $\tau=0.005$.

For IPPO, each of the eight critics is associated with a separate optimizer.
This preserves the independent parameter updates of the decentralized
critics under the Keras 3 optimizer variable-tracking mechanism. CH-MARL and
MAPPO use one centralized critic and one critic optimizer.

The critic target incorporates entropy regularization through
\[
y_t
=
r_t+
\gamma(1-d_t)
\left[
Q_{\mathrm{target}}
(\boldsymbol{s}_{t+1},\boldsymbol{a}_{t+1})
-
\beta\log\pi(\boldsymbol{a}_{t+1}\mid\boldsymbol{s}_{t+1})
\right],
\]
where $d_t$ indicates terminal transitions. The critic minimizes the squared
temporal-difference error,
\[
\mathcal{L}_{Q}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left[
Q(\boldsymbol{s}_b,\boldsymbol{a}_b)-y_b
\right]^2.
\]

The actor objective is
\[
\mathcal{L}_{\pi}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left[
\beta\log\pi(\boldsymbol{a}_b\mid\boldsymbol{s}_b)
-
Q(\boldsymbol{s}_b,\boldsymbol{a}_b)
\right].
\]
For IPPO, the actor value term is obtained from the average of the
agent-specific critics,
\[
Q_{\mathrm{IPPO}}
=
\frac{1}{N}
\sum_{i=1}^{N}
Q_i(\boldsymbol{s}_{i,t},\boldsymbol{a}_{i,t}).
\]
For MAPPO and CH-MARL, the centralized critic directly evaluates the joint
state-action pair.

\subsection{Experimental Protocol and Evaluation Metrics}
\label{subsec:evaluation_metrics}

All three algorithms are evaluated under the same environment configuration
and random seed ($42$). The training history records step-level rewards and
the corresponding episode-level summaries. Importantly, the reported
episode length is based on the actual number of executed steps rather than
the nominal maximum of 100 steps, allowing early termination following
collective target completion.

The primary performance measure is the mean episode reward,
\[
\overline{R}_e
=
\frac{1}{T_e}
\sum_{t=1}^{T_e}r_t,
\]
where $T_e$ denotes the actual number of executed steps in episode $e$.
The corresponding within-episode reward variability is summarized by
\[
SD(R_e)
=
\left[
\frac{1}{T_e-1}
\sum_{t=1}^{T_e}
(r_t-\overline{R}_e)^2
\right]^{1/2}.
\]

In addition to reward, we monitor critic loss, actor loss, policy entropy,
and, for CH-MARL, the transformed equicorrelation parameter $\rho_t$.
The entropy diagnostic is defined as
\[
\mathcal{H}_t
=
-\mathbb{E}
\left[
\log\pi(\boldsymbol{a}_t\mid\boldsymbol{s}_t)
\right].
\]
For CH-MARL, the reported dependence diagnostic is the mean transformed
equicorrelation,
\[
\overline{\rho}_e
=
\frac{1}{T_e}
\sum_{t=1}^{T_e}\rho_t.
\]
Because $\rho_t$ is recorded after the validity-preserving transformation,
the reported value corresponds to the actual correlation parameter used to
construct the Gaussian equicorrelation matrix rather than the unconstrained
network output. For IPPO and MAPPO, no copula dependence parameter is
estimated and the corresponding quantity is therefore not applicable.

For comparative reporting, results are summarized at training milestones
$e\in\{1,5,10,15,20,25\}$. At each milestone, we report mean reward,
reward standard deviation, critic loss, actor loss, policy entropy, and,
where applicable, the mean transformed equicorrelation parameter.

\subsection{Implementation and Reproducibility}
\label{subsec:simulation_reproducibility}

The complete simulation is implemented in \textsf{R} using Keras 3 and
TensorFlow. CPU execution is enforced in the reported implementation to
provide a consistent computational environment. Random-number generation
for both \textsf{R} and TensorFlow is initialized with seed 42. The same
training horizon, batch size, discount factor, learning rates, entropy
coefficient, replay capacity, and environment dynamics are used across all
three algorithms. This common configuration ensures that observed
differences in the learning trajectories can be examined under a controlled
simulation setting rather than arising from different training budgets.