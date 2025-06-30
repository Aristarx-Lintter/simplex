# Probing for Learned Belief Geometry
\label{sec:probing}

We hypothesize that a neural network trained on data from a complex stochastic process does not merely memorize sequences, but learns an internal representation that mirrors the geometry of the optimal Bayesian observer's belief states. To test this, we develop a rigorous probing methodology to determine if a simple, stable linear map exists between the network's activation vectors and the true belief-state geometry of the data-generating process.

\subsection{Methodology: Affine Probing}

Our core method is to search for an affine transformation that maps a network's activation vector $\boldsymbol{a}(w) \in \mathbb{R}^d$ (for a given input context $w$) to a proposed geometric embedding, in this case the true belief state $\boldsymbol{\gamma}(w) \in \mathbb{R}^{d_g}$. If such a map exists, the network's internal representation manifold can be "unwarped" to match the belief geometry. The affine relationship is given by:
\[
\boldsymbol{\gamma}(w) \approx L \boldsymbol{a}(w) + \boldsymbol{b}
\]
where $L \in \mathbb{R}^{d_g \times d}$ is a linear transformation and $\boldsymbol{b} \in \mathbb{R}^{d_g}$ is a bias vector. We formulate this as a weighted least-squares problem to find the optimal transformation $(\boldsymbol{b}, L)$ that minimizes the error across a representative set of contexts.

\subsection{Weighted Least-Squares Formulation}

We assemble a dataset from a set of "anchor" sequences $\{ w_n \}_{n=1}^N$. Let $A \in \mathbb{R}^{N \times (1+d)}$ be the matrix of augmented activation vectors, where the $n$-th row is $[1, \boldsymbol{a}(w_n)]$, and $\Gamma \in \mathbb{R}^{N \times d_g}$ be the matrix of corresponding ground-truth belief vectors $\boldsymbol{\gamma}(w_n)$.

Crucially, not all contexts are equally likely. To ensure our probe reflects the process's true statistics, we weight each anchor sequence by its probability of occurrence, $\Pr(w_n)$. This probability is derived from the ground-truth generative model (see Appendix~A for details). This leads to the objective of finding the transformation $\mathcal{L} \in \mathbb{R}^{(1+d) \times d_g}$ that minimizes the probability-weighted squared error:
\[
\mathcal{L}^* = \argmin_{\mathcal{L}} \mathbb{E}_{w \sim \Pr(w)} \left\| \begin{bmatrix} 1 & \boldsymbol{a}(w) \end{bmatrix} \mathcal{L} - \boldsymbol{\gamma}(w) \right\|_2^2
\]
The solution is given by the normal equations for weighted least squares, solved using a regularized Moore-Penrose pseudoinverse:
\begin{align}
\mathcal{L}^* = (P^{1/2}A)^+ P^{1/2} \Gamma
\end{align}
where $P$ is a diagonal matrix of the context probabilities $\Pr(w_n)$, and the pseudoinverse $(M)^+$ is computed via a truncated SVD. The truncation threshold (regularization `rcond` parameter) is a critical hyperparameter determined via cross-validation.

\subsection{Rigorous Validation via Cross-Validation and Controls}

Finding a low-error map is not sufficient, as high-dimensional spaces can be deceptively flexible. We perform two critical validation steps to ensure the learned geometry is meaningful and not an artifact.

\textbf{1. Hyperparameter Selection via Cross-Validation:} The choice of regularization is critical to avoid overfitting. We determine the optimal regularization parameter (`rcond`) for the pseudoinverse using a 10-fold cross-validation procedure. The dataset of anchor points is partitioned into ten folds. For each candidate `rcond`, we iterate through the folds, training a linear map on nine folds and calculating the prediction error on the held-out tenth. The `rcond` that yields the lowest average error across all ten validation folds is selected. This ensures our model's complexity is tuned based on its ability to generalize to unseen data within the training set. A final, master model is then trained on the *entire* dataset using this optimal `rcond` for maximum accuracy. The results presented in Fig.~\ref{fig:QSlice_details}B are from this final, robustly regularized model.

\textbf{2. Comparison with Control Models:} To falsify the hypothesis that this structure arises "for free" from the architecture's inductive biases, we run the same analysis on control models: (a) randomly initialized, untrained networks of the same architecture, and (b) trained networks with weights shuffled within each layer. As noted in the caption of Fig.~\ref{fig:enter-label}, these controls fail to produce the target geometry, confirming that the structure is learned through training on the data distribution. Together, these tests provide strong evidence that the network learns a specific, non-trivial geometric representation of the task. 