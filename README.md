# SCACM: Semantic Clustered Attention Conditioning for Few-Shot Generation on Unseen Alphabets

Few-shot generative models usually summarize the support images of a new class with a simple **average** of their embeddings. If the examples of a class vary a lot, that average can blur them together.

**SCACM** (Semantic Clustered Attention Conditioning Module) groups the support embeddings with differentiable soft k-means, then lets learned queries attend over the cluster centres and the individual examples. This project builds a **controlled benchmark** to test whether that beats the simple mean inside a few-shot conditional VAE.

## Key results

Omniglot, test characters from **unseen alphabets**, same seed, step count, optimizer and episode sampler for every model. Only the conditioning module changes.

**IWAE negative log-likelihood in nats (lower is better):**

| Conditioning | 1-shot | 5-shot | 10-shot |
|---|:-:|:-:|:-:|
| Mean pooling (baseline) | 109.82 | 106.22 | **105.58** |
| Attention only | **109.81** | **106.14** | 105.69 |
| SCACM (clustered attention) | 110.85 | 106.31 | 105.68 |

**Sample quality (10-shot):**

| Conditioning | SSIM ↑ | 20-way recognizability ↑ | Diversity |
|---|:-:|:-:|:-:|
| Mean pooling | 0.341 | 0.310 | 0.129 |
| Attention only | 0.343 | 0.299 | 0.127 |
| SCACM | 0.340 | 0.294 | 0.130 |
| *Real held-out drawings (ceiling)* | *0.431* | *0.482* | *0.138* |

## Findings

1. **All three conditioning methods are essentially tied.** At 10 shots the gap between the mean and SCACM is 0.1 nats, far smaller than what would matter in practice.
2. **Why the extra machinery does not help here:** Omniglot characters have little variation within a class. Twenty people drawing the same symbol produce similar shapes, so there are no real sub-groups for the clusters to find, and the mean is already a good summary.
3. **This is a useful negative result.** It tells us where *not* to expect gains and points to the data regime where the method should be tested next: classes with clearly different styles, such as fonts or multi-style handwriting.
4. All models are still well below the real-data ceiling (SSIM 0.34 vs 0.43), so the generator itself, not the conditioning, is the main bottleneck at this scale.

## Method

### Few-shot CVAE (shared by all variants)
- CNN support encoder: one embedding per support image
- conditioning module: support embeddings `[B, k, d]` → condition vector `c` (the only part that changes)
- learned conditional prior `p(z | c)`
- amortized posterior `q(z | x, c)`
- Bernoulli decoder `p(x | z, c)`

Training minimizes the negative ELBO, `E_q[−log p(x|z,c)] + β · KL(q(z|x,c) ‖ p(z|c))`, with β warmed up linearly from 0 to 1 over the first 30% of training to reduce posterior collapse.

### The three conditioning modules
| Module | Idea | Parameters |
|---|---|:-:|
| Mean pooling | average of support embeddings | 33.3K |
| Attention only | 2 learned queries, 4 heads, attend over the supports | 132.4K |
| **SCACM** | soft k-means (3 clusters, 3 iterations), centroids start at the support mean plus learned offsets; queries attend over centroids **and** individual supports (a type embedding marks which is which); the mean is kept as a residual | 133.0K |

SCACM is permutation invariant and still works with a single shot. Squared distances are computed directly (no `cdist`), because a one-shot cluster sits exactly on its support point, where the gradient of a Euclidean distance is undefined.

### Evaluation
- **IWAE bound** with 50 importance samples (Burda et al., 2016), tighter than the ELBO.
- **SSIM** between generated samples and real held-out drawings of the same character (batched PyTorch implementation, 7×7 window).
- **20-way recognizability:** whether each generated sample is closest (by SSIM) to its own character among 20 candidates.
- **Diversity:** mean pairwise L1 distance between samples of the same character.
- Fixed evaluation episodes; supports are **nested** (the 1-shot support is part of the 5-shot support, which is part of the 10-shot support) and targets are the same at every shot count.

### Settings
Images 28×28, binarized. Embedding and condition size 128, latent size 32. 8,000 training steps, learning rate 1e-3, gradient clipping 5.0. Shot count drawn uniformly from 1 to 10 at each step, so one model serves all shot settings. Training set: 964 characters from background alphabets; test set: 659 characters from evaluation alphabets.

## How to run

Open the notebook in Google Colab with a GPU runtime and run all cells. Omniglot is downloaded through torchvision and cached after the first load. Each model trains in about 2 minutes on a T4.

## Main libraries

PyTorch · torchvision · NumPy · pandas · Matplotlib

## Limitations

One seed per model. Differences of about 0.1 to 1 nats should be treated as noise until repeated with more seeds.

## Related projects

- [Few-Shot Font Generation (MS thesis)](https://github.com/SadafZehra/Few_Shot_Font_Generation_Using_Conditional_Variational_Autoencoder): the CVAE this module plugs into.
- [Beyond the Mean Prototype](https://github.com/SadafZehra/Beyond-the-Mean-Prototype-Multi-Modal-Classes-in-Meta-Learning): the same clustering idea for few-shot *classification*, where it does give small but significant gains.

## Author

**Sadaf Zehra**, MS Data Science, NED University of Engineering and Technology
[GitHub](https://github.com/SadafZehra) · [LinkedIn](https://www.linkedin.com/in/sadaf-zehra-796858230) · [ORCID](https://orcid.org/0009-0009-7530-9006)
