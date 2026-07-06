\subsection{Data Preparation}
\label{sec:data_preparation}

The dataset generation procedure (Section~\ref{sec:dataset_generation}) produces a
collection of solved PyPSA networks, each containing a 24-step time series of
converged AC power-flow states. Before training, every snapshot of every network is
converted into a single graph sample. This section describes that conversion, which
forms the bridge between the stored datasets and the graph neural network.

\paragraph{Graph topology.}
Each snapshot is represented as a graph in which nodes correspond to buses and edges
correspond to branches (transmission lines and transformers). Every undirected branch
is stored as two directed edges, one in each direction, and the edges are laid out in
an interleaved forward/reverse order
$[\,e_0^{\rightarrow}, e_0^{\leftarrow}, e_1^{\rightarrow}, e_1^{\leftarrow}, \dots\,]$
so that forward edges occupy the even positions. A boolean
\emph{forward-edge mask} records these positions, and a separate
\emph{DC-flow mask} marks the forward \emph{line} edges only (transformers are
excluded from the linearized DC-flow terms, for which they are ill-conditioned).
Transformers are always retained as graph edges to preserve connectivity,
independently of whether they are used for branch-flow supervision.

\paragraph{Node features.}
At each bus $i$ the input feature vector is
\[
x_i = \bigl[\,\mathbb{1}_{\text{slack}},\, \mathbb{1}_{\text{PV}},\, \mathbb{1}_{\text{PQ}},\;
P_i^{\text{in}},\, Q_i^{\text{in}},\, V_i^{\text{ref}},\, \theta_i^{\text{ref}}\,\bigr],
\]
i.e. a three-way one-hot encoding of the bus type followed by the known injections
and reference voltages. Unknown quantities are zero-masked according to bus type: the
active injection is set for PQ and PV buses, the reactive injection only for PQ buses,
the reference magnitude only for PV and slack buses, and the reference angle only for
the slack bus. An optional eighth feature, the normalized generator capacity share
\[
p_i^{\text{nom,share}} = \frac{p_{\text{nom},i}}{\sum_j p_{\text{nom},j}},
\]
is appended when \texttt{use\_pnom\_share=True}, giving either $7$ (default) or $8$
node features. All electrical quantities are expressed in the per-unit system used
during generation; no additional feature standardization is applied.

\paragraph{Edge features.}
Each directed edge carries the branch electrical parameters
\[
a_{ij} = \bigl[\,r_{ij},\, x_{ij},\, b_{ij}^{\text{half}},\, \tau_{ij},\,
g_{ij}^{\text{ser}},\, b_{ij}^{\text{ser}}\,\bigr],
\]
comprising the series resistance and reactance, half the line charging susceptance,
the transformer tap ratio ($\tau_{ij}=1$ for lines), and the series conductance and
susceptance $g^{\text{ser}} + j\,b^{\text{ser}} = 1/(r_{ij} + j\,x_{ij})$. When PTDF
supervision is enabled, the reference PTDF row of a branch may additionally be attached
to its forward edge.

\paragraph{Prediction targets.}
The per-node regression target is the solved bus state
\[
y_i = \bigl[\,V_i^{\ast},\, \theta_i^{\ast},\, P_i^{\ast},\, Q_i^{\ast}\,\bigr] \in \mathbb{R}^{4}.
\]
Auxiliary, variable-length targets are stored alongside each graph for the
physics-informed and distribution-factor losses: the reference DC PTDF matrix
$\Pi^{\ast}$, the sending-end line active and reactive flows $P^{\ast}_{\text{line}}$,
$Q^{\ast}_{\text{line}}$, and, when angle differences are learned, the per-branch angle
targets $\Delta\theta^{\ast}_{ij} = \theta_i^{\ast}-\theta_j^{\ast}$ together with a
breadth-first traversal order used to reconstruct bus angles from branch differences.
Because these tensors have network-dependent shapes, they are excluded from the
standard PyTorch~Geometric mini-batch collation and re-attached per graph.


\subsection{Graph Neural Networks for AC Power Flow Prediction}

\subsubsection{Model Architecture}

\paragraph{Backbone.}
The core architecture is a message-passing graph neural network operating on the
bus--branch graph defined in Section~\ref{sec:data_preparation}. The default backbone
is a Graph Attention Network (GATv2)~\cite{velickovic_graph_2018, brody_how_2022},
chosen for its ability to learn adaptive, edge-conditioned neighbourhood weights
without prior knowledge of the graph structure. GATv2 and the graph
Transformer~convolution both consume the edge feature vector $a_{ij}$ through their
edge-conditioning mechanism; Graph Convolutional
Networks~\cite{kipf_semi-supervised_2017} and \texttt{GraphConv} are also supported as
edge-agnostic alternatives.

The forward computation proceeds as follows:
\begin{enumerate}
    \item \textbf{Node embedding.} Input features are projected to the hidden
    dimension by a linear layer, $h_i^{(0)} = W_{\text{emb}} x_i + b_{\text{emb}}$.
    \item \textbf{Convolution stack.} $L$ message-passing layers are applied. Each
    layer follows a unified block
    \[
        h_i \leftarrow \phi\!\left(\mathrm{Norm}\bigl(\mathrm{Conv}(h_i,\{h_j\}_{j\in\mathcal{N}(i)}, a_{ij})\bigr)\right),
    \]
    with an optional normalization ($\mathrm{Norm}\in\{\text{layer},\text{graph}\}$),
    a configurable activation $\phi$ (LeakyReLU by default; ReLU, GELU, ELU, or SiLU
    optional), optional dropout, and an optional residual connection
    $h_i \leftarrow h_i + h_i^{\text{in}}$.
    \item \textbf{Optional global context.} When enabled, a graph-level embedding
    formed by concatenating mean- and max-pooled node states is projected and added
    back to every node embedding, injecting network-wide context.
    \item \textbf{Output heads.} The final node embeddings feed bus-state prediction
    heads (described below).
    \item \textbf{Edge embeddings for PTDF.} Source and destination node embeddings are
    concatenated with the edge features and passed through an MLP to obtain an edge
    embedding $h_e$, used for the bilinear PTDF prediction.
\end{enumerate}

\paragraph{Angle representation modes.}
The model supports three ways of representing bus voltage angles, selected by
\texttt{angle\_mode}:
\begin{itemize}
    \item \emph{node}: bus angles are predicted directly by a node head, giving a
    node output $[\hat V,\hat\theta,\hat P,\hat Q]\in\mathbb{R}^{4}$.
    \item \emph{edge\_delta}: a dedicated edge head predicts a per-branch angle
    difference $\Delta\hat\theta_{ij}$ for each forward edge; bus angles are then
    reconstructed differentiably by integrating these differences along a
    breadth-first spanning tree rooted at the slack bus
    ($\hat\theta = \texttt{reconstruct\_theta\_from\_delta}(\Delta\hat\theta)$). The
    node head then produces only $[\hat V,\hat P,\hat Q]$.
    \item \emph{both}: node angles and edge angle differences are predicted jointly and
    supervised together.
\end{itemize}
The edge angle head is zero-initialized so that training begins from
$\Delta\hat\theta \approx 0$.

\paragraph{Voltage-magnitude modes.}
With \texttt{vmag\_mode="absolute"} the magnitude head predicts $\hat V$ directly and
its bias is initialized to $1.0$~p.u. With \texttt{vmag\_mode="residual"} the head
predicts a correction around a flat $1.0$~p.u. baseline
($\hat V = 1.0 + \delta\hat V$), with the head zero-initialized so that predictions
start at the flat profile.

\paragraph{Output-head modes.}
Two head configurations are available, selected by \texttt{head\_mode}:
\begin{itemize}
    \item \emph{standard}: four independent linear heads (magnitude, angle, active and
    reactive power), each evaluated on all nodes and selected per bus type in the loss.
    \item \emph{with\_encoder}: three bus-type-specific heads (PQ, PV, slack), each
    applied only to buses of its type and predicting exactly that type's unknowns
    (PQ: $[\hat V,\hat\theta]$; PV: $[\hat\theta,\hat Q]$; slack: $[\hat P,\hat Q]$),
    structurally enforcing the known/unknown split.
\end{itemize}

\paragraph{Bilinear PTDF head.}
From the edge embedding $h_e^{(k)}$ of branch $k$ and node embedding $h_n^{(i)}$ of
bus $i$, a learned bilinear form predicts the Power Transfer Distribution Factor
\[
    \hat{\Pi}_{k,i} = h_e^{(k)\top}\, W_{\text{PTDF}}\, h_n^{(i)},
\]
with $W_{\text{PTDF}}$ a learned weight matrix.

\paragraph{Initialization and outputs.}
Linear layers use Xavier-uniform initialization, with the magnitude and edge-angle
heads given the special initializations described above. The forward pass returns the
node prediction, the predicted PTDF matrix, and (when applicable) the predicted branch
angle differences.

\paragraph{Initial hyperparameters.}
The following default values are used, informed by prior work on power-system
GNNs~\cite{taghizadeh_multi-fidelity_2024} and general graph-learning
practice~\cite{morris_weisfeiler_2021}:
\begin{itemize}
    \item Backbone: GATv2 with $4$ attention heads,
    \item Number of layers: $L = 3$,
    \item Hidden dimension: $64$,
    \item Activation: LeakyReLU (default), with ReLU/GELU/ELU/SiLU as options,
    \item Dropout: disabled by default ($0.1$ when enabled),
    \item Learning rate: $10^{-3}$,
    \item Optimizer: Adam ($\beta_1=0.9$, $\beta_2=0.999$),
    \item Batch size: $1$ for single-graph training, $32$ in the sweep driver.
\end{itemize}


\subsubsection{Physics-Informed Loss}
\label{sec:physics_informed_loss}

The training objective combines supervised regression with physics-informed
constraints. All auxiliary terms are governed by a single configuration object
(\texttt{PhysicsConfig}) that selects which terms are active, their weighting mode
(fixed or adaptive), and their activation schedules.

\paragraph{Supervised loss (masked MSE).}
A masked mean-squared error compares predictions to the PyPSA reference solution,
selecting for each bus only the quantities that are unknown for its type:
\begin{equation}
    \mathcal{L}_{\text{MSE}}
    = \frac{1}{N}\sum_{i=1}^{N} M_i^{(\text{out})} \odot
    \bigl\| \hat{u}_i - u_i^{\ast} \bigr\|^2,
\end{equation}
where PQ buses are supervised on $[\hat V,\hat\theta]$, PV buses on $[\hat\theta,\hat Q]$,
and slack buses on $[\hat P,\hat Q]$. When angle differences are learned
(\texttt{angle\_mode="edge\_delta"}), the node target reduces to $[V,P,Q]$ and an
additional branch-angle term
$\mathcal{L}_{\Delta\theta}=\text{MSE}(\Delta\hat\theta,\Delta\theta^{\ast})$ is added.

\paragraph{Power-balance residual.}
Predicted complex voltages $\hat V_i = \hat v_i e^{j\hat\theta_i}$ are combined with
the network admittance matrix $Y=G+jB$ to obtain the calculated injections
$\hat S_i^{\text{calc}} = \hat V_i\,\overline{(Y\hat V)_i}$. The active-power residual
is enforced at all buses, with the injection taken from the input at PQ and PV buses
and from the prediction at the slack bus:
\begin{equation}
    r_{P,i} = \hat P_i^{\text{calc}} - P_i^{\text{inj}},
    \qquad
    P_i^{\text{inj}} = \begin{cases}
        P_i^{\text{in}} & t_i \in \{\text{PQ},\,\text{PV}\}, \\
        \hat P_i        & t_i = \text{slack}.
    \end{cases}
\end{equation}
The reactive-power residual is likewise enforced at all buses (\emph{full-Q}
formulation), using the known demand at PQ buses and the PyPSA reference $Q_i^{\ast}$ as
supervision at PV and slack buses:
\begin{equation}
    r_{Q,i} = \hat Q_i^{\text{calc}} - Q_i^{\text{inj}},
    \qquad
    Q_i^{\text{inj}} = \begin{cases}
        Q_i^{\text{in}} & t_i = \text{PQ}, \\
        Q_i^{\ast}      & t_i \in \{\text{PV},\,\text{slack}\}.
    \end{cases}
\end{equation}
In the full-Q case the gradient flows through the predicted voltages entering
$\hat Q_i^{\text{calc}}$, not through the output $\hat Q_i$, yielding a
physics-consistent learning signal at all buses. Each residual is normalized by the
corresponding diagonal susceptance to prevent high-injection buses from dominating,
giving the power-balance loss
\begin{equation}
    \mathcal{L}_{\text{PB}}
    = \frac{1}{N}\sum_{i=1}^{N}
    \left[ w_P \Bigl(\tfrac{r_{P,i}}{B_{ii}}\Bigr)^2
         + w_Q \Bigl(\tfrac{r_{Q,i}}{B_{ii}}\Bigr)^2 \right].
\end{equation}
For the \emph{edge\_delta} representation an algebraically equivalent $O(E)$
edge-local form is used, computing per-branch contributions by scatter-reduction over
edges and avoiding the dense $Y$-bus product.

\paragraph{Angle reference penalty.}
An optional soft constraint pins the angle reference by penalizing the mean slack-bus
angle, $\mathcal{L}_{\theta} = \bigl(|\mathcal{S}|^{-1}\sum_{i\in\mathcal{S}}\hat\theta_i\bigr)^2$.

\paragraph{Branch-flow losses.}
Branch-flow supervision is decomposed into four independently controllable terms
formed as the product of a linearization (\emph{DC} or \emph{AC}) and an angle source
(\emph{local}, using the predicted branch difference $\Delta\hat\theta$, or
\emph{global}, using differences of predicted node angles $\hat\theta_i-\hat\theta_j$):
\begin{itemize}
    \item \emph{DC} terms use the linear relation $\hat P_{ij} = \Delta\theta_{ij}/x_{ij}$;
    \item \emph{AC} terms use the full $\pi$-model expressions for $P_{ij}$ and $Q_{ij}$
    from $(\hat V_i,\hat V_j,\Delta\theta_{ij})$ and the branch admittance.
\end{itemize}
Each term is compared against the stored reference line flows and can target active,
reactive, or both components.

\paragraph{PTDF loss.}
The predicted PTDF matrix is supervised against the reference $\Pi^{\ast}$ from PyPSA.
Several modes are available, selected by \texttt{ptdf\_loss\_mode}:
\begin{itemize}
    \item \emph{matrix}: a direct mean-squared error between the predicted and
    reference PTDF matrices;
    \item \emph{flows}: a flow-based loss that maps the predicted PTDF through the bus
    injections to line flows and compares against the reference flows;
    \item \emph{mixed}: a convex combination $\alpha\,\mathcal{L}_{\text{matrix}} +
    (1-\alpha)\,\mathcal{L}_{\text{flows}}$ controlled by \texttt{ptdf\_alpha};
    \item \emph{learned\_edge}: a variant in which a learnable PTDF parameter block is
    trained directly and mapped through injections to line flows;
    \item \emph{stepwise}: a curriculum that trains with the matrix loss up to a
    configurable changepoint epoch (\texttt{ptdf\_changepoint}) and then switches to the
    flow-based loss, so that the model first fits the distribution factors and
    subsequently refines the resulting line flows.
\end{itemize}
The set of supervised branches is controlled by \texttt{ptdf\_branch\_mode}
(lines only, or lines and transformers).

\paragraph{Adaptive weighting and activation schedules.}
Each auxiliary term $\mathcal{L}_a$ may be weighted with a fixed coefficient or,
in \emph{adaptive} mode, rescaled online to a target fraction $\rho_a$ of the current
supervised MSE,
\[
    w_a = \min\!\left(\rho_a\,\frac{\mathcal{L}_{\text{MSE}}}{\mathcal{L}_a + \varepsilon},\; w_a^{\max}\right),
\]
so that physics, PTDF, and flow terms remain commensurate with the supervised signal
throughout training. Each weight is additionally modulated by a per-term activation
schedule that ramps the term on (and optionally off) linearly over a configurable
epoch window, allowing an initial supervised warm-up before physics constraints
engage~\cite{huang_applications_2023}. The total objective is the sum of the supervised
MSE, the branch-angle term, and the weighted physics, PTDF, and branch-flow terms.


\subsubsection{Training Procedure}

\paragraph{Data splitting.}
The available networks are shuffled and partitioned into training, validation, and
test sets in a $70/15/15$ ratio. Each network contributes all of its snapshots as
individual graph samples.

\paragraph{Optimization.}
Model weights are initialized with Xavier-uniform initialization and optimized with
Adam at a default learning rate of $10^{-3}$. A \texttt{ReduceLROnPlateau} scheduler
(halving the learning rate after $10$ epochs without validation improvement) adapts the
step size during training. When a learnable PTDF parameter block is used, it is added as
a separate optimizer parameter group.

\paragraph{Training loop.}
For each epoch the model iterates over mini-batches from the training loader, performs a
forward pass to obtain node predictions, edge embeddings, and (when applicable) branch
angle differences, and computes the combined objective of
Section~\ref{sec:physics_informed_loss}: masked MSE, branch-angle MSE, power-balance and
angle-reference physics terms, PTDF loss, and the four branch-flow terms, each scaled by
its (possibly adaptive and schedule-modulated) weight. Gradients are backpropagated and
the parameters updated with Adam. After each epoch the same objective is evaluated on the
validation set, the scheduler is stepped on the validation loss, and the best-performing
model state is retained. On completion the model checkpoint and a full run-metadata
record --- including per-epoch train/validation traces of every loss component and the
effective adaptive weights --- are saved.

\paragraph{Model output.}
The forward pass returns node predictions of shape $[N,4]$ (or $[N,3]$ under the
\emph{edge\_delta} angle mode, with angles reconstructed from branch differences), a PTDF
prediction of shape $[E, N]$, and, when applicable, per-branch angle differences.


\subsubsection{Post-Processing and Evaluation}

\paragraph{Line-flow calculation.}
Branch power flows are computed from the predicted nodal voltages using the standard
$\pi$-model admittance formulation, yielding sending- and receiving-end active and
reactive flows. When branch angle differences are predicted directly, an equivalent
formulation computes flows from $(\hat V_i,\hat V_j,\Delta\hat\theta_{ij})$ without
explicitly reconstructing bus angles.

\paragraph{Masking and physical constraints.}
At inference, predictions are overwritten with known quantities before any metric is
computed: slack buses use the reference magnitude and a fixed zero angle and keep
predicted $P$ and $Q$; PV buses use the reference magnitude and known active power and
keep predicted angle and reactive power; PQ buses use predicted magnitude and angle and
keep known active and reactive power.

\paragraph{Evaluation metrics.}
Performance is assessed with mean absolute error (MAE) and root-mean-squared error
(RMSE) on voltage magnitude, voltage angle, and active and reactive power, computed per
bus type and aggregated over the test set, together with branch-flow error. Solve,
post-processing, and total inference times are recorded per snapshot.


\subsubsection{Hyperparameter Sweep}

A configurable grid search over the Cartesian product of the training hyperparameters is
used to identify configurations with the best validation performance. In addition to the
classical hyperparameters (depth, width, learning rate, batch size, activation, dropout),
the sweep spans the architectural and loss options introduced above --- backbone type,
angle-representation and voltage-magnitude modes, global pooling, and the physics/PTDF/
flow-loss configurations and their activation schedules. The driver supports
checkpointing and resumption so that large sweeps can be interrupted and continued, and
each configuration is identified by a stable hash of its hyperparameters. Representative
ranges are listed in Table~\ref{tab:hyperparameters}.

\begin{table}[t]
\centering
\caption{GNN hyperparameter search configuration.}
\label{tab:hyperparameters}
\begin{tabular}{p{0.20\textwidth} p{0.16\textwidth} p{0.24\textwidth} p{0.30\textwidth}}
\hline
\textbf{Hyper-parameter} & \textbf{Default} & \textbf{Search range} & \textbf{Reference / notes} \\
\hline
Backbone & GATv2 & \{GATv2, Transformer, GCN, GraphConv\} & \cite{brody_how_2022,kipf_semi-supervised_2017} \\
Number of layers & 3 & \{3, 4, 6\} & Taghizadeh et al.~\cite{taghizadeh_multi-fidelity_2024} \\
Hidden dimension & 64 & \{64, 128\} & Same \\
Learning rate & $10^{-3}$ & $[10^{-4}, 10^{-3}]$ & With ReduceLROnPlateau \\
Activation & LeakyReLU & \{LeakyReLU, ReLU, GELU, ELU, SiLU\} & GELU per Hendrycks \& Gimpel~\cite{hendrycks_gaussian_2023} \\
Dropout & off & $[0.0, 0.5]$ & Srivastava et al.~\cite{srivastava_dropout_2014} \\
Batch size & 32 & \{16, 32, 64\} & Sweep driver default \\
Residual / norm & off / none & \{off, residual\} $\times$ \{none, layer, graph\} & Architectural variants \\
Angle mode & node & \{node, edge\_delta, both\} & Angle representation \\
Vmag mode & absolute & \{absolute, residual\} & Magnitude representation \\
Global pooling & off & \{off, on\} & Graph-context injection \\
Physics fraction $\rho_{\text{phys}}$ & 0.1 & $[0.0, 1.0]$ & Adaptive weighting \\
PTDF fraction $\rho_{\text{PTDF}}$ & 0.1 & $[0.0, 0.5]$ & Adaptive weighting \\
Flow-loss fractions & 0.0 & $[0.0, 0.5]$ per DC/AC $\times$ local/global & Branch-flow terms \\
\hline
\end{tabular}
\end{table}
