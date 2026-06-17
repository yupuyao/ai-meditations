+++
title = "Diffusion Is Not Merely Denoising"
date = "2026-05-08"
+++

# Diffusion Is Not Merely Denoising: An Intuitive View of What DDPM Learns from Sampling

Discussions of diffusion models often begin with ELBOs, score matching, stochastic differential equations, or probability flow ODEs. These derivations are essential, but they can obscure a more basic intuition: what does the model actually learn? Why can a model start from pure Gaussian noise and, after a finite number of sampling steps, gradually produce an image with coherent semantic structure?

This article approaches DDPM from the perspective of sampling. The central claim is simple: a diffusion model does not learn a conventional image denoising filter, nor does it recover an objectively existing "true noise image" during generation. Instead, it learns a family of conditional denoising rules across different noise levels. Together, these rules define a reverse trajectory from a simple Gaussian prior to the complex data distribution.

In this sense, "denoising" in DDPM is better understood as a data-prior-constrained signal-noise decomposition, rather than merely reducing the amplitude of noise.

This article is an intuitive explanation, not a substitute for rigorous probabilistic derivation. I will keep only the minimal formulas needed to align the intuition with the standard DDPM formulation.

---

## Starting from the Sampling Formula: What Does the Model Need to Learn?

The forward process in DDPM can be written as

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon\sim \mathcal{N}(0,I).
$$

This means that any noisy sample \(x_t\) can be viewed as a linear combination of the clean image \(x_0\) and Gaussian noise \(\epsilon\). As \(t\) increases, \(\bar{\alpha}_t\) decreases, the signal component weakens, and the noise component dominates. The signal-to-noise ratio can be written as

$$
\mathrm{SNR}(t)=\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}.
$$

A high SNR means that the image still contains substantial clean signal. A low SNR means that the sample is already close to pure noise.

A common DDPM training objective is

$$
\mathcal{L}=\mathbb{E}_{x_0,\epsilon,t}\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2\right].
$$

At first glance, the model is simply trained to predict Gaussian noise. This is precisely where many beginners become confused: if the model only learns to predict noise, why can it generate images?

A typical reverse sampling step in DDPM can be written as

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t)\right)+\sigma_t z,\quad z\sim \mathcal{N}(0,I).
$$

Ignoring the derivation, this formula performs two conceptual operations.

First, the neural network predicts a noise direction \(\epsilon_\theta(x_t,t)\) from the current sample \(x_t\) and the time step \(t\), and the sample is corrected along that direction.

Second, a new Gaussian noise term \(z\) is injected with an appropriate variance before moving to the next time step.

This already tells us that DDPM sampling is not simply "subtracting a little noise at each step." Instead, at every noise level, the model must decide which variations in the current sample are more consistent with the learned data distribution and which variations should be treated as random components to be removed.

---

## During Sampling, "Noise" Is Not a Unique Visible Object

During training, there is indeed a true noise vector. We start from a clean image \(x_0\), sample \(\epsilon\), and construct \(x_t\). Therefore, the supervision target \(\epsilon\) is known.

Sampling is fundamentally different. Sampling starts from \(x_T\sim\mathcal{N}(0,I)\), without access to any true \(x_0\). Given a sample \(x_t\), we do not know which clean image it came from, nor do we know which part should be interpreted as signal and which part as noise.

This is the key intuition behind diffusion models: during sampling, "noise prediction" should not be understood as finding an objectively existing and uniquely determined noise image inside \(x_t\). Rather, the model estimates a correction direction which, conditioned on \(x_t\) and the noise level \(t\), moves the sample toward the data distribution.

From the perspective of mean squared error, under infinite data and sufficient model capacity, \(\epsilon_\theta(x_t,t)\) learns the conditional expectation

$$
\mathbb{E}[\epsilon\mid x_t,t].
$$

This quantity is closely related to the score of the noise-perturbed data distribution, \(\nabla_{x_t}\log q_t(x_t)\), up to a scale factor. Therefore, predicting noise, predicting the score, and predicting the clean image \(x_0\) are closely connected parameterizations of the same underlying object.

Thus, DDPM is not training a conventional image filter. It is training a conditional vector field: given the current sample and the current noise level, the network estimates the direction in which the sample should move to become more likely under the training data distribution.

---

## Generation Does Not Mean That an Image Was Hidden Inside Noise

A common misunderstanding is that, because diffusion models generate images from pure noise, the final image must somehow already be hidden inside the initial noise.

This is not the right view.

Pure Gaussian noise contains no semantic object. It contains no cat, no face, no building, and no predetermined image content. Generation occurs not because an image was already present in the noise, but because the model has learned a prior over the data distribution and repeatedly uses this prior to interpret the current state.

Given a high-noise sample \(x_t\), there are many possible explanations. It could be interpreted as a noisy version of a face, a noisy version of a cat, or a noisy version of some entirely different image. In the low-SNR regime, this signal-noise decomposition is highly non-unique. The model's role is to select interpretations that are more probable under the training distribution.

If the model is trained only on face images, then during sampling it will tend to interpret certain random structures as early signals that may belong to the face manifold. Components inconsistent with the face prior are gradually suppressed; components consistent with the face prior are gradually reinforced. The final generated samples are therefore usually faces, not cats or cars.

Therefore, it is more accurate to say that the model uses a learned data prior to gradually choose a path from noise to image, rather than saying that it discovers a pre-existing image inside the noise.

This is one of the most interesting aspects of diffusion generation: at every step, the model faces an underdetermined signal-noise decomposition problem, and the sampling trajectory is the accumulated result of a sequence of conditional interpretations.

---

## The Time Step Is Not Auxiliary Information; It Defines the Task

Many beginners regard the time embedding as an engineering trick. Since the noisy image already contains information about the noise level, why explicitly tell the network which step it is processing?

This underestimates the role of time conditioning in DDPM. The time step, or more generally the noise level, is not a minor auxiliary variable. It defines the conditional denoising problem that the model is solving.

The same sample \(x_t\) should be interpreted differently depending on its assumed noise level. If it is treated as a high-noise sample, the model should be conservative in deciding which structures are reliable. If it is treated as a low-noise sample, the model can preserve edges, textures, and local details more aggressively. In other words, the time condition informs the model of the current SNR and determines the scale at which signal-noise decomposition should be performed.

At high noise levels, the model relies more heavily on global, low-frequency, category-level priors. A weak structure may be interpreted as the rough outline of a face or the body layout of an animal.

At intermediate noise levels, the model begins to stabilize object shapes, boundaries, part relations, and spatial structure.

At low noise levels, the model mainly refines texture, color, edges, and local consistency.

Therefore, time conditioning does not merely control a "denoising strength." It specifies the interpretation scale appropriate for the current SNR. Without time conditioning, a single shared network would be forced to solve all denoising tasks across all noise levels simultaneously, leading to an averaged prediction that may have acceptable global MSE but large bias at individual noise levels.

Strictly speaking, the essential component is not necessarily a discrete time-step embedding. What matters is conditioning on the noise level. This can be represented by the discrete time step \(t\), the noise standard deviation \(\sigma\), the log-SNR, or other continuous noise parameterizations. The key requirement is that the model must know which perturbed distribution it is currently operating on.

---

## Why Add New Noise During Reverse Sampling?

The DDPM reverse update not only subtracts a predicted noise direction, but also injects fresh Gaussian noise. This seems counterintuitive: if the goal is to generate a clean image, why add noise while denoising?

The reason is that DDPM learns a probabilistic reverse process, not a deterministic image enhancement procedure. The forward process gradually destroys structure from \(x_0\) to \(x_T\). The reverse process must recover the data distribution from a simple prior distribution. To correctly match the reverse transition distribution \(p_\theta(x_{t-1}\mid x_t)\), the reverse process usually needs a variance term.

The newly injected noise serves two purposes.

First, it preserves stochasticity in sampling and therefore supports diversity. Under the same condition or category, the model can generate different samples rather than collapsing to a single average image.

Second, it keeps the scale of the reverse transition compatible with the forward diffusion process. Noise injection is not meant to damage generation; it helps keep the sample near the noise-level distribution on which the model was trained.

However, not every diffusion sampler must explicitly inject random noise. DDIM shows that one can construct a deterministic non-Markovian sampling process under the same training objective. The score-based SDE framework also shows that diffusion models can be sampled either through stochastic differential equations or through the corresponding deterministic probability flow ODE.

Thus, stochastic noise is not the only source of generation. The essential object is the learned conditional score, or conditional denoising vector field. Stochasticity mainly affects sampling paths, sample diversity, and how the reverse distribution is represented.

---

## Why Standard DDPM Usually Requires Multiple Sampling Steps

A standard DDPM usually cannot generate high-quality samples from Gaussian noise in a single reverse step. This statement should be interpreted carefully: it refers to a standard DDPM without additional distillation or special training. Later methods such as progressive distillation, consistency models, and shortcut models can achieve one-step or few-step generation, but they modify the training objective or introduce additional distillation mechanisms.

Why does standard DDPM need multiple steps?

Intuitively, the transformation from a Gaussian noise distribution to the real data distribution is too complex to be represented as a single simple update. The reverse process is not a linear mapping; it is a trajectory following the score field of the data distribution. Each step solves a local conditional denoising problem: at the current noise level, in which direction should the sample move?

If the step size is too large, the sampler is effectively using a crude numerical approximation of a complex trajectory. The model is then forced to extrapolate on states far away from its training distribution, and errors accumulate rapidly. Generated images may suffer from structural collapse, inconsistent textures, or semantic artifacts.

A useful intuition is that each DDPM step must keep the input within the model's "comfort zone" for that noise level. During training, the model sees samples drawn from a specific perturbed data distribution at each noise level. If sampling jumps too far, the current sample may no longer resemble the training distribution at the target time step, and the model's denoising estimate becomes unreliable.

Therefore, multi-step sampling is not required because the model is weak. It is required because the difficult generation problem has been decomposed into many local, learnable, and stable conditional estimation problems.

---

## Why Diffusion Models Can Skip Steps

If multi-step sampling is important, why can DDIM, DPM-Solver, EDM, and other samplers substantially reduce the number of sampling steps?

The key point is that the discrete number of DDPM steps is not fundamental. It is a discretization of an underlying continuous, or approximately continuous, reverse dynamics. If we can approximate this reverse trajectory more accurately, we can use fewer network evaluations.

DDIM constructs a non-Markovian reverse process that enables faster sampling while using the same training objective. DPM-Solver treats sampling as solving a diffusion ODE and designs higher-order numerical solvers specialized for diffusion dynamics. EDM further emphasizes the separation of noise schedules, network preconditioning, loss weighting, and sampler design.

From an intuitive perspective, step skipping works only if the sample after the jump remains in a region that the model can reliably interpret. If the skipped trajectory still lands near a reasonable distribution at the lower noise level, the model can continue sampling. If the jump is too aggressive, numerical integration error and model extrapolation error both increase.

Thus, the central problem of efficient sampling is: how can we use as few neural network evaluations as possible while still moving stably along the correct reverse trajectory?

---

## Why Diffusion Models Are Usually Less Prone to GAN-Style Mode Collapse

The original GAN formulation trains a generator and discriminator through a minimax game. GANs can generate sharp images, but their training dynamics are difficult, and mode collapse is a classic failure mode: the generator may cover only a small subset of the data distribution while ignoring other modes.

Diffusion models are generally less prone to this specific GAN-style mode collapse for three reasons.

First, the DDPM objective resembles supervised regression. The model receives noisy samples at different time steps and predicts noise, score-related targets, or clean-image-related targets. This objective does not depend on a dynamic adversarial game between a generator and a discriminator, and is therefore often easier to optimize stably.

Second, diffusion models learn the local geometry of the data distribution across many noise levels. High-noise stages capture coarse distributional structure, while low-noise stages capture fine detail. This multi-scale denoising objective decomposes complex generation into many conditional estimation problems.

Third, stochastic sampling and score-field estimation help cover multiple modes. Especially in unconditional or weakly conditional generation, different initial noises and different stochastic paths can lead to different samples.

However, this does not mean that diffusion models are immune to diversity loss. Insufficient model capacity, biased training data, too few sampling steps, overly strong conditioning, or an excessively large classifier-free guidance scale can all reduce diversity. A more precise statement is that diffusion models are typically less vulnerable to classical training-induced GAN mode collapse, but they still face distribution coverage, sampling bias, and diversity-fidelity trade-offs.

---

## Why More Training Can Improve Samples Even After the Loss Nearly Converges

In diffusion training, it is common to observe that the MSE loss decreases slowly after a certain point, while generated sample quality continues to improve. This is not surprising.

The total MSE is an aggregate statistic. It averages errors across time steps, image regions, frequency components, and semantic modes. Even if the total loss changes only slightly, the model may still be improving score estimates in visually important regimes.

Small errors at low noise levels directly affect edges, texture, and color consistency. Errors at intermediate noise levels affect object shapes and part relations. Errors at high noise levels can affect global layout and semantic category. Since a sampling chain consists of many steps, errors at one step propagate to later states. Visual quality is therefore sensitive not only to the average error, but also to where the error occurs.

Another relevant intuition comes from the spectral bias of neural networks: many networks tend to learn low-frequency, global, smoother function components before learning high-frequency and local details. This does not mean that late-stage diffusion training merely learns high-frequency noise. Rather, later improvements may concentrate on visually sensitive details, texture consistency, and low-probability modes.

Therefore, similar MSE values do not necessarily imply similar sample quality. Generation quality depends on the distribution of errors, not only their mean.

---

## Can Low-Pass Filtering Help Sampling?

From an image-processing perspective, one may ask: if early sampling should first establish low-frequency structure, can we directly add low-pass filtering during sampling?

This is an interesting idea, but it must be treated cautiously. A conventional low-pass filter can suppress high-frequency fluctuations and make an image appear smoother. However, the denoising direction in DDPM is not a simple frequency filter. It is an estimate of the data distribution's score. Some high-frequency components are noise, but others are valid texture, edges, or local structures. Blind low-pass filtering may improve early visual stability, but it may also destroy the correct score direction.

More generally, any handcrafted filtering operation must answer a specific question: does it keep the sample near the distribution learned by the model at the current noise level? If the filtering operation moves the sample away from the training distribution, the denoiser may produce incorrect corrections in subsequent steps.

Therefore, low-pass filtering can be explored as a heuristic experiment, but it should not be regarded as an equivalent replacement for DDPM sampling. Modern diffusion samplers more commonly improve stability and efficiency through better noise schedules, preconditioning, time discretization, and ODE/SDE solver design.

---

## A Mental Picture of the Entire Sampling Process

The DDPM sampling process can be summarized as

$$
\text{pure noise }x_T
\rightarrow
\text{coarse semantic selection}
\rightarrow
\text{structural formation}
\rightarrow
\text{detail refinement}
\rightarrow
\text{clean image }x_0.
$$

More concretely:

At high noise levels, the SNR is low and the image is almost invisible. The model relies primarily on data priors to determine coarse category, layout, and global structure.

At intermediate noise levels, the SNR increases. The model stabilizes object boundaries, part relations, and spatial structure.

At low noise levels, the image is already mostly formed. The model mainly refines texture, color, edges, and local consistency.

This process is not about extracting a pre-existing image from noise. It is about following the probability landscape defined by the learned data distribution and gradually moving a random sample toward high-density regions.

---

## Conclusion: DDPM Learns a Family of Conditional Denoising Rules

The key strength of DDPM is not merely that it can denoise. Its strength is that it decomposes a difficult generative modeling problem into many stable conditional denoising subproblems.

From the training perspective, the model learns

$$
(x_t,t)\mapsto \epsilon_\theta(x_t,t),
$$

that is, how to estimate the random component that should be removed at a given noise level.

From the score-based perspective, the model learns the gradient field of the noise-perturbed data distribution,

$$
\nabla_{x_t}\log q_t(x_t).
$$

From the sampling perspective, the model learns a reverse trajectory from a Gaussian prior to the data distribution.

Therefore, "denoising" in diffusion models should not be understood as conventional image denoising. It is better understood as conditional probabilistic modeling: at every noise level, the model uses the learned data prior to reinterpret the current sample and estimate a correction direction toward high-density regions of the data distribution.

This explains why time conditioning matters, why multi-step sampling is useful, why better samplers can skip steps, why diffusion models are often more stable than GANs, and why generation quality may continue to improve even when the aggregate loss changes only slightly.

In one sentence:

A diffusion model does not discover an image hidden inside noise; it uses a learned data prior to gradually interpret noise as an image.

Learning is easy; easy to learn is hard. The elegance of DDPM is that it reformulates a difficult generation problem into a sequence of conditional denoising tasks that are easier to supervise, easier to optimize, and easier to scale.

---

## References

[1] Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., & Ganguli, S. "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." ICML 2015.  
PMLR: <https://proceedings.mlr.press/v37/sohl-dickstein15.html>  

[2] Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.  
NeurIPS: <https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html>  

[3] Song, Y., & Ermon, S. "Generative Modeling by Estimating Gradients of the Data Distribution." NeurIPS 2019.  
NeurIPS: <https://papers.nips.cc/paper/9361-generative-modeling-by-estimating-gradients-of-the-data-distribution>  

[4] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.  
OpenReview: <https://openreview.net/forum?id=PxTIG12RRHS>  

[5] Song, J., Meng, C., & Ermon, S. "Denoising Diffusion Implicit Models." ICLR 2021.  
OpenReview: <https://openreview.net/forum?id=St1giarCHLP>  

[6] Nichol, A. Q., & Dhariwal, P. "Improved Denoising Diffusion Probabilistic Models." ICML 2021.  
PMLR: <https://proceedings.mlr.press/v139/nichol21a.html>  

[7] Dhariwal, P., & Nichol, A. "Diffusion Models Beat GANs on Image Synthesis." NeurIPS 2021.  
OpenReview: <https://openreview.net/forum?id=AAWuCvzaVt>  

[8] Ho, J., & Salimans, T. "Classifier-Free Diffusion Guidance." NeurIPS 2021 Workshop / arXiv 2022 version.  
arXiv: <https://arxiv.org/abs/2207.12598>

[9] Karras, T., Aittala, M., Aila, T., & Laine, S. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022.  
NeurIPS: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html>  

[10] Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps." NeurIPS 2022.  
NeurIPS: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html>  

[11] Salimans, T., & Ho, J. "Progressive Distillation for Fast Sampling of Diffusion Models." ICLR 2022.  
OpenReview: <https://openreview.net/forum?id=TIdIXIpzhoI>  

[12] Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. "Consistency Models." ICML 2023.  
PMLR: <https://proceedings.mlr.press/v202/song23a.html>  

[13] Rahaman, N., Baratin, A., Arpit, D., et al. "On the Spectral Bias of Neural Networks." ICML 2019.  
PMLR: <https://proceedings.mlr.press/v97/rahaman19a.html>  

[14] Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. "Generative Adversarial Networks." NeurIPS 2014.  
arXiv: <https://arxiv.org/abs/1406.2661>

[15] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.  
CVF: <https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html>  
