---
title: "Do Llamas Work in English? On the Latent Language of Multilingual Transformers"
authors: [Chris Wendler, Veniamin Veselovsky, Giovanni Monea, Robert West]
affiliation: EPFL
arxiv: "2402.10588v4 [cs.CL], 8 Jun 2024"
code: https://github.com/epfl-dlab/llm-latent-language
note: "Converted from PDF to Markdown. Math rewritten in LaTeX notation; figures replaced by descriptions of their content, since the plots themselves don't survive conversion."
---

# Do Llamas Work in English? On the Latent Language of Multilingual Transformers

## Abstract

We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language — a question of key importance for understanding how language models function and the origins of linguistic bias. Focusing on the Llama-2 family of transformer models, our study uses carefully constructed non-English prompts with a unique correct single-token continuation. From layer to layer, transformers gradually map an input embedding of the final prompt token to an output embedding from which next-token probabilities are computed. Tracking intermediate embeddings through their high-dimensional space reveals three distinct phases, whereby intermediate embeddings:

1. start far away from output token embeddings;
2. already allow for decoding a semantically correct next token in middle layers, but give higher probability to its version in English than in the input language;
3. finally move into an input-language-specific region of the embedding space.

We cast these results into a conceptual model where the three phases operate in "input space", "concept space", and "output space", respectively. Crucially, our evidence suggests that the abstract "concept space" lies closer to English than to other languages, which may have important consequences regarding the biases held by multilingual language models.

**Code and data:** https://github.com/epfl-dlab/llm-latent-language

---

## 1 Introduction

Most modern LLMs are trained on massive corpora of mostly English text (Touvron et al., 2023; OpenAI, 2023). Despite this, they achieve strong performance on a broad range of downstream tasks, even in non-English languages (Shi et al., 2022). This raises a compelling question: how are LLMs able to generalize so well from their mainly English training data to other languages?

Intuitively, one way to achieve strong performance on non-English data in a data-efficient manner is to use English as a **pivot language**: translate input to English, process it in English, then translate the answer back. This works well when implemented explicitly (Shi et al., 2022; Ahuja et al., 2023; Huang et al., 2023). The guiding inquiry here is whether pivoting to English also occurs **implicitly** when LLMs are prompted in non-English.

Many in the research community and popular press assume the answer is yes, epitomized by claims such as, "The machine, so to say, thinks in English and translates the conversation at the last moment into Estonian" (Piir, 2023). This work moves beyond such speculation and investigates the question empirically.

**Why it matters.** On the one hand, implicitly using English as an internal pivot could bias LLMs toward Anglocentric patterns — lexicon, grammar, metaphors — while also shaping deeper behaviors related to emotional stance (Boroditsky et al., 2003) or temporal reasoning (Núñez and Sweetser, 2006). On the other hand, if LLMs do *not* use English as a pivot, it raises the question of how else they work so well in low-resource languages.

**Why it's hard.** After the input layer, transformers do not operate on discrete tokens but on high-dimensional floating-point vectors. Determining whether those vectors correspond to English, Estonian, Chinese — or to no language at all — is an open problem, and to the authors' knowledge the pivot-language question had not been addressed empirically before.

### Summary of contributions

- Applying the **logit lens** (Nostalgebraist, 2020) — the unembedding operation applied prematurely at intermediate layers — already decodes a contextually appropriate token early on, giving a limited glimpse at the model's internal state.
- Prompts are carefully devised so that a logit-lens-decoded token can be checked for both semantic correctness and language membership (e.g. translating French *fleur* → Chinese 花).
- Tracking language probabilities across layers: no contextually appropriate tokens are decoded in the first half of layers, followed by a sudden shift of probability mass onto the English version ("flower"), and finally a shift to the correct target-language token (花).
- Analyzing latents directly as high-dimensional Euclidean points suggests that in middle layers the transformer operates in an abstract **"concept space"** partially orthogonal to a language-specific **"token space"**, reached only in the final layers. Under this interpretation, proximity to English tokens follows from an *English bias in concept space*, rather than from the model translating to English and "restarting" its forward pass.

---

## 2 Related work

### Multilingual language models

Multilingual LMs are trained to handle multiple input languages simultaneously: mBERT (Devlin et al., 2018), mBART (Liu et al., 2020), XLM-R (Conneau et al., 2020a), mT5 (Xue et al., 2021), XGLM (Lin et al., 2022), mGPT (Shliazhko et al., 2022), BLOOM (Scao et al., 2022), PolyLM (Wei et al., 2023). Frontier models (GPT-4, PaLM, Llama-2) perform better in English due to Anglocentric training data (Huang et al., 2023; Bang et al., 2023; Zhang et al., 2023) but still do well across languages (Shi et al., 2022).

Methods for transferring LM capabilities across languages include aligning contextual embeddings (Schuster et al., 2019; Cao et al., 2020), relearning embedding matrices during finetuning (Artetxe et al., 2020), or doing so repeatedly during pretraining (Chen et al., 2023).

Several approaches leverage English as an *explicit* pivot: Zhu et al. (2023) augment Llama with multilingual instruction-following via its English representations; Zhu et al. (2024) fine-tune on translation data plus English-only instructional data to enhance multilingual reasoning; Husain et al. (2024) show romanized + English data helps non-Latin low-resource languages. Prompting strategies also pivot through English — translating prompts to English first (Shi et al., 2022; Ahuja et al., 2023; Etxaniz et al., 2023) or instructing English chain-of-thought (Huang et al., 2023). Using high-resource languages can, however, bias generation in low-resource languages, e.g. grammatically (Papadimitriou et al., 2022).

For encoder-only models like mBERT, converging evidence suggests a **language-agnostic space in later layers** following language-specific early layers (Libovický et al., 2020; Conneau et al., 2020b; Muller et al., 2021; Choenni and Shutova, 2020).

### Mechanistic interpretability

MI aims to reverse-engineer neural networks via circuit discovery (Nanda et al., 2023; Conmy et al., 2023), controlled task-specific training (Li et al., 2022; Marks and Tegmark, 2023), and causal tracing (Meng et al., 2022; Monea et al., 2023). Sparse probing (Gurnee et al., 2023) has revealed monosemantic French and German language neurons and context-dependent German n-gram circuits (Quirke et al., 2023).

The most relevant tools are:

| Tool | Mechanism |
|---|---|
| **Logit lens** (Nostalgebraist, 2020) | Applies the LM head prematurely at earlier layers, no additional training |
| **Tuned lens** (Belrose et al., 2023) | Trains an affine map so intermediate states mimic final-layer predictions |
| **Direct logit attribution** (Elhage et al., 2021) | Generalizes logit lens to per-attention-head logit contributions |

> **Why logit lens and not tuned lens:** the tuned lens is specifically trained to map internal states — even English-corresponding ones — to the final non-English prediction. Its optimization criterion would "optimize away" the very signal of interest.

---

## 3 Materials and methods

### 3.1 Language models: Llama-2

Llama-2 (Touvron et al., 2023) was trained on a multilingual corpus dominated by English (89.70% of the corpus). Given two trillion training tokens, even small percentages are large in absolute terms (0.17% = 3.4B German tokens; 0.13% = 2.6B Chinese tokens). Llama-2 is therefore considered multilingual despite its English bias.

**Versions.** All three sizes are studied, with 8-bit quantization (Dettmers et al., 2022):

| Model | Parameters | Layers | Embedding dim $d$ | Vocab $v$ |
|---|---|---|---|---|
| Llama-2-7B | 7B | 32 | 4096 | 32,000 |
| Llama-2-13B | 13B | 40 | 5120 | 32,000 |
| Llama-2-70B | 70B | 80 | 8192 | 32,000 |

**Architecture.** Autoregressive, decoder-only, residual-based transformer. Shape is preserved throughout the forward pass: one latent vector per input token $x_1,\dots,x_n \in V$. Initial latents $h^{(0)}_1,\dots,h^{(0)}_n \in \mathbb{R}^d$ come from a learned embedding dictionary. Each latent is updated layer by layer by adding a residual:

$$h^{(j)}_i = h^{(j-1)}_i + f_j\!\left(h^{(j-1)}_1, \dots, h^{(j-1)}_i\right) \tag{1}$$

where $f_j$ (a transformer block) is a masked self-attention layer followed by a feed-forward layer, with a residual connection and RMS normalization in between (Vaswani et al., 2017; Touvron et al., 2023). **Due to RMS normalization, all latents lie on a $d$-dimensional hypersphere of radius $\sqrt{d}$.**

For prediction, the final latent is multiplied by the unembedding matrix $U \in \mathbb{R}^{v \times d}$, giving logits $z_i = U h^{(m)}_i \in \mathbb{R}^v$, converted to probabilities $P(x_{i+1} = t \mid x_1,\dots,x_i) \propto e^{z_{it}}$ via softmax.

### 3.2 Interpreting latent embeddings: logit lens

Since latents have the same shape in all layers, any latent can be turned into a token distribution by treating it as if it were a final-layer latent. This yields one next-token distribution $P(x_{i+1} \mid h^{(j)}_i)$ per position $i$ and layer $j$.

> **Figure 1.** Logit lens applied to Llama-2-7B on a translation prompt ending with `Français: "fleur" - 中文: "`. Each cell shows the top next token per position (x-axis) and layer (y-axis). The final layer correctly ranks 花 (translation of *fleur*) on top, whereas intermediate layers decode the English "flower". Cell color indicates entropy of the next-token distribution (low = blue, high = red). Plotting tool: Belrose et al. (2023).

### 3.3 Data: tasks for eliciting latent language

The logit lens maps latents to token distributions, but a mapping from token distributions to *languages* is still needed. Many tokens are ambiguous — "an" occurs in English, French, and German. To circumvent this, prompts $x_1 \dots x_n$ are constructed so the correct next token $x_{n+1}$ is (1) obvious and (2) unambiguously attributable to one language.

#### Prompt design

Three text-completion tasks (Chinese used as the example target language):

**Translation task** — translate the preceding non-English word into the target language. Four demonstration pairs, then a fifth word without its translation (中文 means "Chinese"):

```
Français: "vertu" - 中文: "德"
Français: "siège" - 中文: "座"
Français: "neige" - 中文: "雪"
Français: "montagne" - 中文: "山"
Français: "fleur" - 中文: "
```

**Repetition task** — simply repeat the last word instead of translating it:

```
中文: "德" - 中文: "德"
中文: "座" - 中文: "座"
中文: "雪" - 中文: "雪"
中文: "山" - 中文: "山"
中文: "花" - 中文: "
```

**Cloze task** — predict a masked word in a sentence. Given a target word, an English sentence starting with the word is generated by prompting GPT-4, the target word is masked, and the sentence is translated to the other languages. Two demonstrations are sampled from the remaining words:

```
A "___" is used to play sports like soccer and basketball. Answer: "ball".
A "___" is a solid mineral material forming part of the surface of the earth. Answer: "rock".
A "___" is often given as a gift and can be found in gardens. Answer: "
```

#### Word selection

To enable unambiguous language attribution, a closed set of words per language is constructed. Chinese is a particularly clean case: many single-token words, no spaces. Llama-2's vocabulary is scanned for single-token Chinese words (mostly nouns) that have a single-token English translation, so probabilities for the correct Chinese word and its English analog can be read directly off the next-token distribution.

For robustness, all experiments are also run on German, French, and Russian. Selected Chinese/English words are translated, and for each language, words sharing a token prefix with the English version are discarded (they would render language detection ambiguous).

**Final counts:** 139 Chinese, 104 German, 56 French, 115 Russian words (see Appendix A.1).

### 3.4 Measuring latent language probabilities

The logit lens is applied to the latents $h^{(j)}_n$ of the **last input token** $x_n$ at each layer $j$, obtaining $P(x_{n+1} \mid h^{(j)}_n)$ per layer. Since single-token words are selected in both Chinese (ZH) and English (EN), the probability of language $\ell \in \{\text{ZH}, \text{EN}\}$ is defined as the probability of the next token being $\ell$'s version $t_\ell$ of the correct single-token word:

$$P(\text{lang} = \ell \mid h^{(j)}_n) := P(x_{n+1} = t_\ell \mid h^{(j)}_n)$$

Note this does **not** define a distribution over languages, since generally $\sum_\ell P(\text{lang} = \ell) < 1$. For other languages (and corner cases in Chinese and English), multiple tokenizations and whitespaces must be accounted for (Appendix A.2).

---

## 4 Results

Section 4.1 takes a probabilistic view via the logit lens for all tasks and model sizes (results are consistent across languages, so Chinese is the focus; see Appendix B for French, German, Russian). Section 4.2 takes a geometric view of how latents drift layer by layer.

### 4.1 Probabilistic view: logit lens

> **Figure 2.** Language probabilities during the Llama-2 forward pass. Rows = (a) translation task from the union of German/French/Russian into Chinese, (b) Chinese repetition task, (c) Chinese cloze task. Columns = model sizes 7B / 13B / 70B. x-axis: layer index. y-axis: logit-lens probability of the correct Chinese next token (blue) or its English analog (orange). Error bars: 95% Gaussian CIs over input texts (353 for translation, 139 for repetition and cloze). An entropy heatmap runs above each plot.

**Translation and cloze tasks** (consistent across model sizes):

1. Neither the correct Chinese token nor its English analog gets noticeable probability mass during the **first half** of layers.
2. Around the **middle** layer, English rises sharply, then declines.
3. Chinese grows slowly, crosses over English, and **spikes in the last five layers**.

**Repetition task:** Chinese rises alongside English (discussed in §6). This is in contrast to all other languages, where English rises first (Appendix B).

**Entropy:** high in the first half of layers while both $P(\text{lang}=\text{ZH})$ and $P(\text{lang}=\text{EN})$ are near zero; a sharp drop coincides with the rise of $P(\text{lang}=\text{EN})$; entropy then stays low, with a slight rebound as mass shifts from English to Chinese. With $32{,}000 \approx 2^{15}$ vocabulary tokens, the early entropy of ~14 bits implies a close-to-uniform next-token distribution (~15 bits).

#### Path visualization

Figure 2 only considers two tokens. To get an intuition for the entire distribution, dimensionality reduction is used. Distance between a latent $h_n$ and a token $t$ is defined via negative log-likelihood under the logit lens:

$$d(h_n, t) = -\log P(x_{n+1} = t \mid h_n)$$

Classical multidimensional scaling then embeds tokens and latents in an approximately distance-preserving joint 2D space. (Intra-token and intra-latent distances are set to $\max_{h,t} d(h,t)$, acting as a "spring force" pushing 2D points apart.)

> **Figure 3.** Latent trajectories through transformer layers (German→Chinese translation, 70B). Latents (○) and output tokens (×) embedded in 2D via MDS; latents for the same prompt connected by a rainbow path from layer 1 (red) to 80 (violet). Correct Chinese next tokens labeled in blue, English analogs in orange. **Takeaway: latents reach the correct Chinese token after a detour through English.**

Two observations:

1. An English and a Chinese **token cluster** emerge — the same latent gives high probability to an entire language, not just the language-specific version of the correct token.
2. Paths **first pass through the English cluster**, and only later reach the Chinese cluster.

Together: when translating a German word to Chinese, Llama-2 takes a "detour" through an English subspace.

### 4.2 Geometric view: an 8192D space odyssey

Focus: one task (translation), one model size (70B, $d = 8192$).

#### Embedding spheres

Output token embeddings (rows of $U$) and latents $h$ cohabitate the same $d$-dimensional space. Due to RMS normalization, latents lie on a hypersphere of radius $\sqrt{d} \approx 90.1$. Analyzing the 2-norm of output token embeddings (mean 1.52, SD 0.23) shows they also approximately lie on a sphere, of radius 1.52.

#### Token energy

Token embeddings occupy their sphere unevenly: the first 25% of principal components account for 50% of total variance, the first 54% for 80%.[^pc]

Intuition: consider a hypothetical extreme case where tokens lie in a proper subspace ("token subspace") of the full $d$-dimensional space (empirically $U$ has rank $d$, so output embeddings actually span all of $\mathbb{R}^d$). If a latent $h$ has a component orthogonal to the token subspace, that component is irrelevant for predicting the next token from $h$ alone — logits are scalar products of latent and token vectors. The orthogonal component can still matter for later layers' computations, **but the logit lens is blind to it.**

A latent's angle with the token subspace thus measures how much of it is irrelevant for immediately predicting the next token. The mean squared cosine between $h$ and the token embeddings, normalized by the mean squared cosine among token embeddings themselves,[^norm] gives the squared **token energy**:

$$E(h)^2 = \frac{\frac{1}{v}\lVert \hat{U}h \rVert_2^2 \big/ \lVert h \rVert_2^2}{\frac{1}{v^2}\lVert \hat{U}\hat{U}^\top \rVert_F^2} = \frac{v}{d}\,\frac{\lVert \hat{U}h \rVert_2^2}{\lVert \hat{U}\hat{U}^\top \rVert_F^2} \tag{2}$$

where $\hat{U}$ is $U$ with 2-normalized rows. This captures $h$'s proximity to token subspace relative to a random token's proximity to it.

**Result:** RMS token energy is low (~20%) and mostly flat before layer 70, then **suddenly spikes — exactly when next-token predictions switch from English to Chinese.**

> **Figure 4.** Anatomy of the transformer forward pass when translating to Chinese: layer-by-layer evolution of (a) entropy, (b) token energy, (c) language probabilities; (d) a 3D sketch of latents traveling on a hypersphere (actual space is 8192D). 甜 means "sweet".

#### The three phases

| Phase | Layers (70B) | Entropy | Token energy | Language |
|---|---|---|---|---|
| **Phase 1** | 1–40 | High (14 bits, near-uniform) | Low | None dominates |
| **Phase 2** | 41–70 | Low (1–2 bits) | Low | English dominates |
| **Phase 3** | 71–80 | Low | High (20% → 30%) | Chinese dominates |

[^pc]: Moreover, Cancedda (2024) showed a significant fraction of the principal components can be omitted as long as attention sinking is preserved.
[^norm]: In practice $\hat{U}^\top\hat{U}$ is used instead of $\hat{U}\hat{U}^\top$ in (2) — equal Frobenius norm, more efficient to compute.

---

## 5 Conceptual model

The transformer's job is essentially to map the input embedding of the current token to the output embedding of the next token.

**Phase 1** builds a better feature representation of the *current* token from its input embedding: dealing with tokenization issues (integrating preceding tokens of the same word), integrating words into larger semantic units, etc. Not yet directly concerned with predicting the next token — latents remain largely orthogonal to output token space (low token energy), giving small dot products with output token embeddings and thus high entropy.

**Phase 2** places latents in an abstract **"concept space"** that is no longer orthogonal to output token space. Latent "concept embeddings" sit closer to those output token embeddings that can express the respective concept (across languages, synonyms, etc.), giving low entropy. Among concept-relevant tokens, **English variants lie closer to the concept embedding than non-English variants** — due to the model's overwhelming English exposure during training — hence higher English probabilities. Concept embeddings still carry much information beyond output tokens (input-specific context, target-language information), so token energy remains low.

**Phase 3** maps abstract concepts to concrete words/tokens in the target language. Information irrelevant for next-token prediction is discarded → **spike in token energy**.

### Sketch (Figure 4d)

A strongly simplified toy picture in 3D rather than the actual 8192D:

- All embeddings (output tokens and latents) lie on a sphere around the origin.
- Token embeddings lie on the **equator**, spread mostly along the **x-axis** (left/right), which captures **language** (English left, Chinese right).
- The **y-axis** (front/back) captures **concepts** — here along a 1D "sweetness" scale.
- The **z-axis** (bottom/top) is an **extra degree of freedom** storing context, language, etc.

A forward pass moves along the sphere's surface: the latent starts at the north pole (Phase 1), orthogonal to both output-token and concept embeddings; Phase 2 rotates it into concept space, where English tokens are more likely because their embeddings have a stronger concept component $y$; Phase 3 rotates it along the equator into the target language's hemisphere, onto the output token that best captures the active concept in that language.

---

## 6 Discussion

Latent embeddings do lie further from the correct next token in the input language than from its English analog, giving overwhelmingly English internal representations as seen through the logit lens. It might be tempting to conclude that Llama-2 uses English as an implicit pivot, similar to prior *explicit* uses of English as a pivot (Shi et al., 2022; Ahuja et al., 2023; Huang et al., 2023).

**But the answer must be more nuanced.** Much of the latents' "energy" points in directions largely orthogonal to output token embeddings and thus does not matter for next-token prediction. The model can use these directions as extra degrees of freedom for building rich feature representations (Yosinski et al., 2014, 2015; Geva et al., 2022) — an abstract "concept space". Under this interpretation:

> The model's internal lingua franca is **not English but concepts — concepts that are biased toward English.** English could still be seen as a pivot language, but in a *semantic* rather than a purely *lexical* sense.

**On the repetition task.** The translation and cloze tasks operate at a semantic level; word repetition is purely syntactic. Yet in most languages the pattern is similar, with tokens first going through an "English phase" — possibly because recognizing that the task is to copy a token requires semantic understanding, achieved only in concept space, which is closer to English token embeddings.

That said, the English-first pattern is **less pronounced** on repetition, where the input language rises earlier or (for Chinese) even simultaneously with, or faster than, English. This may be due to **tokenization**: 100% single-token words were chosen for Chinese, versus only 13% for Russian, 43% for German, 55% for French (Table 1). **Where language-specific tokens are available, the detour through English seems less pronounced.** This supports prior concerns about tokenization, which not only burdens minority languages with more tokens per word (Artetxe et al., 2020) but also forces latents through an English-biased semantic space.

**Future work** should investigate in what ways an English bias in latent space could be problematic, e.g. by biasing downstream behavior — designing experiments building on psycholinguistics, which has shown that concepts may carry different emotional values in different languages (Boroditsky et al., 2003) and that colexification may affect cognition (Di Natale et al., 2021). It should also study how English bias changes when English dominance is reduced during training, e.g. by applying the method to Llama-2 derivatives with a different language mix (Goddard, 2023; Plüster, 2023; Huang, 2023; Kim, 2023), or by using less Anglocentric tokenizers.

---

## Limitations

- **Model family.** Focus on Llama-2 limits claims about other English-dominated models (but see Appendix B.2 for initial evidence that Mistral-7B behaves identically). The method relies on model parameters, so little can be said about closed-source models. The methods generalize straightforwardly to other autoregressive transformers and, given parameters, to non-autoregressive ones.
- **Task simplicity.** The tasks are simple and toy-like, providing a highly controlled context. Essential as a first step to illustrate existence, but future work should extend to a wider range of tasks — culturally sensitive problems, popular use cases, and analyses beyond single tokens.
- **Concept space structure.** Limited understanding of the structure of this space in its original high-dimensional form. Better mapping it out is an important future direction.
- **Logit lens noise.** The logit lens grants only approximate access to internal beliefs about the output at a given position. Everything else in the intermediate representations (information to construct keys, queries, values, or intermediate calculations not contributing to output beliefs) remains hidden and enters the analysis only as noise.

---

## Appendix A — Additional methodological details

### A.1 Word translation

English words were translated to French, German, and Russian using DeepL, translating both the individual words and their cloze sentences; word context from the cloze task was included to disambiguate homonyms. Translations were then filtered to remove words sharing a prefix token across English and the target language (e.g. French *photographier* for "photograph" shares the `photo` prefix token). Cloze translations where the target word didn't align with the expected word from the individual word translation (DeepL failures) were also filtered. Hence differing final word counts per language.

**Table 1 — Aggregated translation task dataset sizes**

| Language | Total | Single token |
|---|---|---|
| de | 287 | 126 |
| fr | 162 | 88 |
| ru | 324 | 45 |
| zh | 353 | 353 |

**Table 2 — Repetition task dataset sizes** (identical to Table 3, cloze task)

| Language | Total | Single token |
|---|---|---|
| de | 104 | 45 |
| en | 132 | 132 |
| fr | 56 | 31 |
| ru | 115 | 15 |
| zh | 139 | 139 |

**Table 4 — Translation statistics between languages** (rows = source, columns = target; total, with single-token counts in brackets)

| | de | en | fr | ru | zh |
|---|---|---|---|---|---|
| **de** | – | 120 (120) | 56 (31) | 105 (15) | 120 (120) |
| **en** | 104 (45) | – | 57 (31) | 114 (15) | 132 (132) |
| **fr** | 93 (40) | 118 (118) | – | 104 (15) | 118 (118) |
| **ru** | 90 (41) | 114 (114) | 49 (26) | – | 115 (115) |
| **zh** | 104 (45) | 132 (132) | 57 (31) | 115 (15) | – |

### A.2 Computing language probabilities

Llama-2's vocabulary is searched for all tokens that could be the **first token** of the correct word in the respective language — all prefixes of the word both without and with a leading space. For Chinese and Russian, tokenizations based on UTF-8 encodings of unicode characters are also considered. For language $\ell$ with target word $w$:

$$P(\text{lang} = \ell) := \sum_{t_\ell \in \text{Start}(w)} P(x_{n+1} = t_\ell) \tag{3}$$

where $\text{Start}(w)$ is the set of starting tokens of $w$.

**Example.** If the correct Chinese word is 花 ("flower"), tokenizable either as the single token 花 or via its UTF-8 encoding `<0xE8>·<0x8A>·<0xB1>`:

```
P(lang = ZH) = P("花") + P("<0xE8>")

P(lang = EN) = P("f") + P("fl") + P("flow") + P("_f") + P("_fl")
             + P("_flo") + P("_flow") + P("_flower")
```

(all token-level prefixes of "flower" and "_flower", where `_` represents a leading space).

---

## Appendix B — Additional results

Results for all languages: Chinese, English, French, German, Russian.

**Language probability plots** (with entropy heatmaps): aggregated translation task in Fig. 5, repetition in Fig. 7, cloze in Fig. 9. Individual language pairs for translation in Figs. 11, 13, 15, 17, 19.

> The same pattern holds across **almost all languages and model sizes**: noise in the early layers, English in the middle, target language at the end. **The only exception is the Chinese repetition task.**

**Energy plots:** aggregated translation in Fig. 6, repetition in Fig. 8, cloze in Fig. 10; individual pairs in Figs. 12, 14, 16, 18, 20. All consistent with the theory in §5.

### B.1 Low-resource language: Estonian

Analysis with Llama-2-7B on Estonian (Fig. 21). Its low-resource status is already evident in tokenization: **only 1 of 99 Estonian words is representable as a single token.**

| Task | Finding |
|---|---|
| **Copy** | Behaves most similarly to Chinese — Estonian probability exceeds English already in intermediate layers |
| **Translation** | Final-layer success probability is much lower than for the main-paper languages, but the same effect appears: intermediate logit-lens distributions concentrate on correct English tokens, transitioning to Estonian only in the final layers |
| **Cloze** | Appears too hard — 0% success probability after the last layer, possibly due to extremely low Estonian resources in the training data. Interestingly, success probability is slightly above 0% in intermediate layers, when the logit lens decodes to English. Might improve with synonyms or human-authored cloze examples instead of GPT-4 |

### B.2 Other models: Mistral

Analysis on Mistral-7B (Fig. 22) is consistent with Llama-2, pointing at the universality of the findings.

---

## References

- Ahuja, K., et al. 2023. *MEGA: Multilingual evaluation of generative AI.*
- Artetxe, M., Ruder, S., Yogatama, D. 2020. *On the cross-lingual transferability of monolingual representations.* ACL.
- Bang, Y., et al. 2023. *A multitask, multilingual, multimodal evaluation of ChatGPT on reasoning, hallucination, and interactivity.* arXiv:2302.04023.
- Belrose, N., et al. 2023. *Eliciting latent predictions from transformers with the tuned lens.* arXiv:2303.08112.
- Biderman, S., et al. 2023. *Pythia: A suite for analyzing large language models across training and scaling.* ICML.
- Boroditsky, L., Schmidt, L. A., Phillips, W. 2003. *Sex, syntax, and semantics.* In *Language in Mind*, MIT Press.
- Cancedda, N. 2024. *Spectral filters, dark signals, and attention sinks.* arXiv:2402.09221.
- Cao, S., Kitaev, N., Klein, D. 2020. *Multilingual alignment of contextual word representations.*
- Chen, Y., et al. 2023. *Improving language plasticity via pretraining with active forgetting.*
- Choenni, R., Shutova, E. 2020. *What does it mean to be language-agnostic?* arXiv:2009.12862.
- Conmy, A., et al. 2023. *Towards automated circuit discovery for mechanistic interpretability.* arXiv:2304.14997.
- Conneau, A., et al. 2020a. *Unsupervised cross-lingual representation learning at scale.*
- Conneau, A., et al. 2020b. *Emerging cross-lingual structure in pretrained language models.* ACL.
- Dettmers, T., et al. 2022. *LLM.int8(): 8-bit matrix multiplication for transformers at scale.* arXiv:2208.07339.
- Devlin, J., et al. 2018. *BERT: Pre-training of deep bidirectional transformers for language understanding.*
- Di Natale, A., Pellert, M., Garcia, D. 2021. *Colexification networks encode affective meaning.* Affective Science 2(2).
- Elhage, N., et al. 2021. *A mathematical framework for transformer circuits.* Transformer Circuits Thread.
- Etxaniz, J., et al. 2023. *Do multilingual language models think better in English?*
- Geva, M., et al. 2022. *Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space.*
- Goddard, C. 2023. *llama-polyglot-13b.* HuggingFace.
- Gurnee, W., et al. 2023. *Finding neurons in a haystack: Case studies with sparse probing.* arXiv:2305.01610.
- Huang, B. 2023. *vigogne-2-13b-instruct.* HuggingFace.
- Huang, H., et al. 2023. *Not all languages are created equal in LLMs: Improving multilingual capability by cross-lingual-thought prompting.*
- Husain, J. A., et al. 2024. *RomanSetu: Efficiently unlocking multilingual capabilities of LLMs via romanization.*
- Kim, D. 2023. *Llama-2-ko-DPO-13B.* HuggingFace.
- Li, K., et al. 2022. *Emergent world representations.* arXiv:2210.13382.
- Libovický, J., Rosa, R., Fraser, A. 2020. *On the language neutrality of pre-trained multilingual representations.* arXiv:2004.05160.
- Lin, X. V., et al. 2022. *Few-shot learning with multilingual generative language models.* EMNLP.
- Liu, Y., et al. 2020. *Multilingual denoising pre-training for neural machine translation.* TACL 8.
- Marks, S., Tegmark, M. 2023. *The geometry of truth.* arXiv:2310.06824.
- Meng, K., et al. 2022. *Locating and editing factual associations in GPT.* NeurIPS 35.
- Monea, G., et al. 2023. *A glitch in the matrix? Locating and detecting language model grounding with Fakepedia.* arXiv:2312.02073.
- Muller, B., et al. 2021. *First align, then predict.* EACL.
- Nanda, N., et al. 2023. *Progress measures for grokking via mechanistic interpretability.* arXiv:2301.05217.
- Nostalgebraist. 2020. *Interpreting GPT: The logit lens.* LessWrong.
- Núñez, R. E., Sweetser, E. 2006. *With the future behind them.* Cognitive Science 30(3).
- OpenAI. 2023. *GPT-4 technical report.*
- Papadimitriou, I., Lopez, K., Jurafsky, D. 2022. *Multilingual BERT has an accent.*
- Piir, R. 2023. *Finland's ChatGPT equivalent begins to think in Estonian as well.* ERR News.
- Plüster, B. 2023. *LeoLM.* laion.ai.
- Quirke, L., et al. 2023. *Training dynamics of contextual n-grams in language models.* arXiv:2311.00863.
- Radford, A., et al. 2019. *Language models are unsupervised multitask learners.* OpenAI blog.
- Rimsky, N. 2023. *Decoding intermediate activations in Llama-2-7b.* LessWrong.
- Scao, T. L., et al. 2022. *BLOOM: A 176B-parameter open-access multilingual language model.* arXiv:2211.05100.
- Schuster, T., et al. 2019. *Cross-lingual alignment of contextual word embeddings.* NAACL.
- Shi, F., et al. 2022. *Language models are multilingual chain-of-thought reasoners.*
- Shliazhko, O., et al. 2022. *mGPT: Few-shot learners go multilingual.*
- Touvron, H., et al. 2023. *Llama 2: Open foundation and fine-tuned chat models.* arXiv:2307.09288.
- Vaswani, A., et al. 2017. *Attention is all you need.* NeurIPS 30.
- Wei, J., et al. 2022. *Chain-of-thought prompting elicits reasoning in large language models.* NeurIPS 35.
- Wei, X., et al. 2023. *PolyLM: An open source polyglot large language model.*
- Xue, L., et al. 2021. *mT5: A massively multilingual pre-trained text-to-text transformer.* NAACL.
- Yosinski, J., et al. 2014. *How transferable are features in deep neural networks?* NeurIPS 27.
- Yosinski, J., et al. 2015. *Understanding neural networks through deep visualization.* arXiv:1506.06579.
- Zhang, X., et al. 2023. *Don't trust ChatGPT when your question is not in English.* EMNLP.
- Zhu, W., et al. 2023. *Extrapolating large language models to non-English by aligning languages.*
- Zhu, W., et al. 2024. *Question translation training for better multilingual reasoning.*

### Acknowledgements (abridged)

Thanks to Nina Rimsky (2023) for the Llama-2 wrapper and logit lens implementation; Lucia Quirke for MI and experimental-setup input; Saibo Geng for help with the Chinese dataset; and to Nicola Cancedda, David Garcia, Eric Horvitz, Manoel Horta Ribeiro, Maxime Peyrard, Tim Davidson, Valentin Hartmann, and Zachary Horvitz for discussion. West's lab is partly supported by SNSF (200021_185043, TMSGI2_211379), Swiss Data Science Center (P22_08), H2020 (952215), and gifts from Meta, Google, and Microsoft.
