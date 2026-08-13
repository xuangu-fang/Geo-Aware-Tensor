# Shared refocus: geometry-aware tensor learning

This note supersedes the earlier requirement that the two papers be maximally
orthogonal or win only on deliberately difficult regimes.  Both papers share a
single scientific core:

> Classical Bayesian or neural tensor decomposition becomes geometry-aware when
> each mode factor is a function on its own geometric domain, represented and
> regularized by that mode's operator, while the multilinear CP/Tucker structure
> remains explicit and empirically necessary.

The two papers may share datasets, notation, and basic geometry components. They
differ in the inferential object—Bayesian posterior versus deterministic neural
factor adaptation—not in whether they are tensor models.

## Non-negotiable tensor structure

For an order-`M` partially observed tensor, retain its mode indices rather than
flattening all coordinates into one regression table:

\[
  Y_{i_1\ldots i_M},\qquad (i_1,\ldots,i_M)\in\Omega.
\]

The common Tucker form is

\[
  \widehat Y_{i_1\ldots i_M}
  =\langle \mathcal G,
    f^{(1)}(x^{(1)}_{i_1})\otimes\cdots\otimes
    f^{(M)}(x^{(M)}_{i_M})\rangle,
\]

and CP is its superdiagonal-core special case.  Each mode has its own domain,
operator, eigenpairs, factor rank, and geometry metadata:

\[
  \mathcal A_m\phi_{mk}=\lambda_{mk}\phi_{mk},\qquad
  f^{(m)}(x)=W_m^\top\Phi_m(x).
\]

A method is not eligible for the central claim if it collapses this model to a
single joint feature vector followed by ordinary regression, or if an unrestricted
coordinate MLP can bypass every multilinear factor.

## Paper A: Bayesian geometry-aware tensor decomposition

The simplest eligible model is a Bayesian spectral Tucker/CP model:

\[
  W_{m,kr}\mid\alpha_{mr},p_m
  \sim\mathcal N\!\left(0,
  [\alpha_{mr}(1+\lambda_{mk})^{p_m}]^{-1}\right),
\]

\[
  \mathcal G_{r_1\ldots r_M}\mid\eta_{r_1\ldots r_M}
  \sim\mathcal N(0,\eta^{-1}_{r_1\ldots r_M}),
  \qquad Y_\Omega\mid W,\mathcal G\sim
  \mathcal N(P_\Omega\widehat Y,\sigma^2I).
\]

Rank/band ARD should prune components or core interactions.  Posterior
uncertainty must arise from the tensor factors/core, not only from a flat joint
Gaussian feature model.  A structured variational posterior, alternating
Gaussian updates, Laplace approximation, or exact small-core conditional is
acceptable if its approximation is stated.

Minimum causal ablations:

1. ordinary Bayesian CP/Tucker with index or unconstrained factors;
2. geometry-aware factors with the correct operators;
3. identical tensor model with wrong/permuted operators;
4. flat operator GP with no low-rank tensor factorization;
5. rank/core ARD removed while keeping parameter budget comparable.

## Paper B: neural geometry-aware tensor decomposition

The simplest eligible neural model retains the multilinear decoder:

\[
  f^{(m)}(x)=W_m^\top
  \big[s_{\theta_m}(\Lambda_m,e_m)\odot\Phi_m(x)\big]
  +\epsilon_m r_{\theta_m}(\Phi_m(x)),
\]

\[
  \widehat Y(x_1,\ldots,x_M)
  =\langle\mathcal G,
  f^{(1)}(x_1)\otimes\cdots\otimes f^{(M)}(x_M)\rangle.
\]

The spectral/phase adapter may be nonlinear, but it must operate inside a mode
factor.  Any residual is band-limited or norm-controlled so it cannot silently
replace the tensor model.

Minimum causal ablations:

1. classical CP/Tucker;
2. raw-coordinate Neural CP/Tucker;
3. geometry-aware Neural CP/Tucker;
4. identical architecture with wrong/permuted mode geometry;
5. geometry-aware monolithic INR with no multilinear decoder;
6. core/rank/band adapter ablations.

## Data representation and masks

- Store tensor values, per-mode coordinates, per-mode operator metadata, and a
  boolean mask separately.  Flattening is allowed only inside a likelihood or
  minibatch sampler and must preserve mode-index tuples.
- Synthetic generators should be independent of the fitted decoder whenever the
  experiment is a robustness or confirmation claim.  Matched CP/Tucker draws are
  useful only for identifiability and implementation checks.
- Include simple regimes where low-rank multilinearity is genuinely plausible;
  difficulty is not a virtue by itself.
- Public arrays such as time×x×y Active Matter and cylinder PIV should remain
  tensors.  Their boundary/topology metadata define mode geometry; unavailable
  geometry must be stated rather than invented.
- Every mask/noise draw is shared across models.  Primary rates may include
  0.5%, 1%, 2%, and 5%; not every paper needs the same hardest rate.

## Iteration discipline

Each paper performs at least four refocus rounds.  Every round records:

1. the smallest proposed formula change;
2. which tensor/geometry hypothesis it tests;
3. the data tensor and mask protocol;
4. a cheap falsifying pilot;
5. the measured result and failure diagnosis;
6. exactly one or two justified component changes for the next round.

Do not add architectural machinery merely to repair an unfavorable number.  A
round is successful only if its ablations show both of the following:

- correct geometry contributes beyond an equal-capacity wrong geometry; and
- explicit tensor factorization contributes beyond a geometry-aware flat model.

The existing Bayesian operator-field and intrinsic-phase INR results remain
useful precursor evidence.  They are not, by themselves, evidence for the new
shared claim because their final predictors are not explicit Bayesian/neural
tensor decompositions.
