# JEPA / latent-prediction SSL, collapse, and collapse diagnostics: verified survey

Survey date: 2026-09-30. Bib file: `jepa_ssl.bib` (83 entries).
How entries were checked: titles and full author lists come from the arXiv API. Venues were confirmed against the venue's own listing: CVF open access, ECVA (ECCV), proceedings.neurips.cc, PMLR, OpenReview (ICLR/TMLR/NeurIPS 2025/ICML 2026), or Crossref. For every claim marked *(read)* I read the paper text myself; the claim is not taken from an abstract summary. Anything marked **[our inference]** is my reasoning about LeLunar, not a claim made in the cited paper.

---

## Closest prior work to our collapse findings (honest assessment)

### 1. The "position-code" collapse has been seen before. It has not been characterized in our setting.

- **CAPI** (`darcet2025cluster`, TMLR 2025) names and describes **"positional collapse"** *(read, Sec. 3.1 + App. F)*: "the positional encoding started outweighing the content of the patch embeddings. In the extreme case, the model learns to predict the position of the masked tokens instead of their content, resulting in a zero loss with a trivial model. In most observed cases, both content and position information are entangled in the target representation." Key details:
  - It was intermittent ("does not arise in all training runs").
  - Partially collapsed early models still reached ~60% ImageNet linear accuracy.
  - It was diagnosed only qualitatively, from PCA maps of patch features.
  - The fix was to run Sinkhorn-Knopp **separately at each position**, so the cluster targets carry zero mutual information with position.

  **This is the prior art we must cite as the first description of the phenomenon.** CAPI differs from us in four ways:
  - it is single-modal;
  - its targets are clustering assignments trained with cross-entropy, not continuous layer-normed EMA features trained with smooth-L1;
  - it gives no quantitative detector;
  - it does not analyze which standard monitors fail.

  CAPI settled on RoPE, but the paper does not say which position encoding the collapsed runs used.
- **JEPAs focus on slow features** (`sobal2022joint`) *(read)*: a JEPA trained with VICReg or SimCLR can minimize the loss by encoding only fixed background noise. The authors show that a trivial solution which copies Gaussian noise satisfies the objective. This is the same class of failure as ours: a signal that is trivially predictable, carries no content, and still satisfies the regulariser. In our case grid position plays the role of the slow feature **[our inference]**.
- **Context prediction** (`doersch2015unsupervised`) *(read)*: nearest-neighbour retrievals "match patches from the same absolute location in the image, regardless of content", caused by a chromatic-aberration shortcut. This is the earliest cross-sample signature of an absolute-position shortcut in SSL.
- **StoP** (`bar2024stochastic`) *(read)*: in I-JEPA, "adjacent tokens tend to share similar features, implying a correlation between the features and spatial location". StoP "prevents overfitting to locations features" by using stochastic mask-token positions. The authors also had to stop the model from "collapse[-ing] back to deterministic positional embeddings", which suggests the optimiser actively pushes toward exact positional information **[our inference]**.
- **SiamJEPA** (`yamada2026siamjepa`) *(read)*: the Random Shuffle Teacher permutes the teacher's tokens so "the student cannot reliably predict the target representation based solely on positional or local spatial information". A linear probe recovers patch grid position from healthy JEPA tokens at **94.3%** accuracy (18.7% with RST). So being able to decode position does not mean the model has collapsed. What matters is whether sample content is also present, and a cross-sample test measures exactly that.
- **What DINO saw** (`pawlowsky2026dino`) *(read)*: SSL ViTs, including the RoPE-based DINOv3 and MAE, have output channels that are "almost purely positional ramp functions regardless of input image". Supervised ViTs do not. So RoPE does not remove absolute position from SSL features.
- **Absolute position without absolute positional embeddings.** CPVT (`chu2023conditional`) *(read)*: "the zero paddings here are important to make the model be aware of the absolute positions". Islam et al. and Kayhan & van Gemert show CNNs encode absolute position through padding and boundary effects. Our conv stems plus 2D RoPE can therefore still hand the model an absolute position code **[our inference]**.
- DVT, Registers, SINDER and test-time registers describe position-linked or high-norm **artifacts** layered on top of content. That is a different phenomenon from collapse.

**Assessment.** I found no paper that describes position-code collapse in any of these settings:
- (a) JEPA or latent regression with continuous targets;
- (b) cross-modal prediction;
- (c) runs where loss, within-tile effective rank, within-tile token cosine gap and VICReg var/cov all look healthy while the cross-sample gap is about 0.

The defensible claim is: "first quantitative characterization and detection of positional collapse in a continuous-target, cross-modal JEPA, and a demonstration that standard monitors are blind to it". "First observation" is not defensible, because CAPI got there first.

### 2. Cross-sample gap diagnostic: the principle exists, this exact form was not found

- **LiDAR** (`thilak2024lidar`) *(read)* uses "clean samples as surrogate classes": it takes the LDA rank of between-sample versus within-sample scatter. It is the closest established monitor in spirit.
- **Latent MIM** (`wei2024towards`) *(read, Table 1)* uses the average pairwise cosine similarity between **mean-pooled latents of different samples** as its collapse indicator. The value is 1.00 for their collapsed naive model and 0.50 with momentum targets. For a *pure* position code, pooled embeddings are identical across samples, so a pooled cross-sample metric would flag it **[our inference]**. A position-dominated but *partial* collapse is less visible after pooling.
- Other cross-sample precedents:
  - alignment/uniformity (`wang2020understanding`);
  - InfoNCE retrieval (`oord2018representation`);
  - SimSiam's per-channel std of ℓ2-normalized outputs (`chen2021exploring`, *(read)*);
  - DenseCL's dense negatives, which are pooled features of other images (`wang2021dense`, *(read)*).
- **Bi-Orthogonal Factor Decomposition** (`doshi2026biorthogonal`) splits token activations into orthogonal positional and content factors with an ANOVA-style decomposition. The analogous quantity for a detector would be the fraction of target variance explained by the per-position mean over samples, a principled complement to our gap **[our inference/suggestion]**.
- **Not found:**
  - a JEPA monitor defined as cos(pred, own target) − cos(pred, another sample's target at the same position);
  - an explicit statement that within-image metrics (rank, token-cos gap, var/cov over tokens) cannot see a position code.

  Within-image objectives could in principle be satisfied by a position code **[our inference; not reported by the authors]**. Examples are Latent MIM's PatchDisc, an InfoNCE over the patches of the *same* image, and its inter-patch similarity constraint.
- Practical note: every cross-sample test needs at least 2 samples per step. At B=1 (grid 128) it has to be computed across steps or ranks.

### 3. VICReg / SIGReg findings versus the literature

- **C-JEPA** (`mo2024connecting`) *(read)* adds VICReg to I-JEPA with a small weight (β_vicreg = 0.001). In their pseudo-code, std and cov act on context embeddings. They report better stability. Our results complicate a naive transfer: the variance hinge on predictions is binding and harmful, and the covariance term favours the position code. **No paper found reports the variance hinge on JEPA predictions as harmful.**
- **VICRegL** (`bardes2022vicregl`) *(read)* keeps a global (pooled) VICReg criterion alongside the local terms: "The global criterion is sufficient for the vectors to not collapse to trivial solutions." A pooled per-sample variance is exactly the statistic that penalizes a pure position code. Token-level var/cov computed across positions can be satisfied by variation across positions alone **[our inference]**.
- **Implicit variance regularization** (`halvagal2023implicit`): non-contrastive EMA/predictor dynamics already regularize variance implicitly. That is one possible reason an explicit hinge would be redundant or harmful **[hypothesis]**.
- **SIGReg** (`balestriero2025lejepa`) *(read)* is an Epps-Pulley test on random 1-D slices (default 256), applied to per-view *global* embeddings. The paper bounds the per-sample gradient as |∂EP/∂z_i| ≤ 4σ²/N.
  - **VISReg** (`wu2026visreg`) reports that SIGReg "suffer[s] from vanishing gradients under collapse", and keeps a VICReg-style variance term.
  - Le MuMo JEPA and CR-JEPA apply SIGReg to pooled or retrieval embeddings; LeWorldModel applies it to world-model latents. All three report it works.
  - No paper found reports SIGReg as inert in a dense, cross-modal JEPA. Our result is consistent with VISReg's critique and with LeJEPA's O(1/N) per-sample gradient bound **[our interpretation]**.

### 4. Dimensional collapse while the loss keeps falling

- Dimensional collapse is defined in `hua2021feature` and `jing2022understanding`.
- `li2022understanding` *(read)*: a partial dimensional collapse and the accuracy drop it causes are "not visible in any of the loss metrics".
- RankMe motivates its rank monitor by JE-SSL's "non informative loss values".
- MoCo v3 (`chen2021empirical`): ViT-SSL instability "can be hidden by apparently good results".
- Attention entropy collapse (`zhai2023stabilizing`) is a separate collapse that goes with training instability; it may be related to the grad-norm explosion we see, but we have not tested this.
- **Not found:** a JEPA paper that ties the drop in effective rank specifically to a grad-norm explosion.

### 5. What we borrow, verified

- **V-JEPA 2.1** (`murlabadia2026vjepa`) *(read)*:
  - dense loss L = L_predict + L_ctx;
  - context weight λ_i = λ/√d_min(i, M), where d_min is the distance in blocks to the closest masked token;
  - deep self-supervision at **four** levels (three intermediate blocks plus the output), with both losses applied at each level;
  - the context loss raises ADE20K linear mIoU from 22.2 to 33.9;
  - rationale: it prevents "visible tokens from acting as global aggregators".
- data2vec *(read)* averages the normalized outputs of the top-K teacher blocks as its target.
- DSN is the classic deep-supervision reference.
- DINOv3's Gram anchoring fixes dense-feature degradation over long schedules. That is another dense-only failure that global metrics miss.

---

## Could NOT verify / caveats

1. `sobal2022joint`: the NeurIPS 2022 SSL-workshop acceptance appears only in the arXiv comment. The workshop's accepted-papers page was unreachable, and OpenReview has only the CoRR record. Cited as arXiv.
2. `roy2007effective`: venue and year verified via the Zenodo record (EUSIPCO 2007, Poznan). Page numbers (606-610 in a secondary search snippet) were **not** verified and are omitted.
3. `agrawal2022alpha` (α-ReQ) has no arXiv version under this title. It was verified on the NeurIPS 2022 proceedings page. arXiv 2202.05808 ("Investigating Power laws in Deep Representation Learning", same authors) is a different paper.
4. `lei2025m3jepa`: PMLR lists 8 authors; arXiv v6 lists 11 (adding Kun Fan, Zhonglin Jiang, Yong Chen). The bib uses the PMLR list.
5. `assran2025vjepa`: arXiv metadata splits "Mojtaba Komeili" into two names. The bib uses the OpenReview/DBLP spelling and truncates to 12 authors plus "and others" (29 total).
6. `assran2023self`: the arXiv comment wrongly says "International Conference on Computer Vision". CVF open access lists it in **CVPR 2023**.
7. `darcet2025cluster`: the TMLR listing spells the title "Cluster and Predict Latent**s** Patches…". The bib uses the arXiv title.
8. `zhou2022ibot`: the ICLR 2022 OpenReview title drops the "iBOT:" prefix. `shwartzziv2023information`: the NeurIPS title is "An Information Theory Perspective…" while arXiv has "Information-Theoretic". The bib uses the NeurIPS title.
9. `simeoni2026dinov3`: arXiv 2508.10104 (2025), accepted to TMLR (OpenReview, published 2026-05-15). Cited as TMLR 2026; switch to arXiv 2025 if preferred.
10. `vanassel2025joint`, `jiang2025vision`: the NeurIPS 2025 main-track proceedings were not yet listed at proceedings.neurips.cc (only Creative AI). Venue verified via OpenReview (NeurIPS 2025 spotlight). `huang2026text`: ICML 2026 verified via OpenReview; PMLR pages not checked.
11. arXiv-only as of 2026-09-30, with no venue found: LeJEPA, V-JEPA 2, V-JEPA 2.1, MC-JEPA (OpenReview shows it as a rejected ICLR 2024 submission), SALT, DMT-JEPA, LeWorldModel, VISReg, SiamJEPA, CR-JEPA, GeoJEPA, What DINO saw, BFD, Cookbook, CPC.
12. X-JEPA's claim to be the first JEPA beyond unimodal domains is the authors' own claim and is not verified. M3-JEPA (arXiv Sep 2024) predates it.
13. Absence claims ("not found") come from targeted searches of arXiv, OpenReview and the web, plus reading the closest papers. They cannot rule out a paper I did not find.

---

## Entries

Format for each entry: **key**, what the paper does, how it relates to LeLunar, and how it was verified.

### JEPA family

- **assran2023self** (I-JEPA, CVPR 2023)
  - What it does: predicts EMA-target features of masked target blocks from a context block, using a narrow ViT predictor fed positional mask tokens.
  - Relation: our base recipe. We extend it to cross-modal targets, target-modality and sun-geometry conditioning, deep supervision and a dense context loss.
  - Verified: arxiv.org/abs/2301.08243; openaccess.thecvf.com/CVPR2023 listing.
- **bardes2024revisiting** (V-JEPA, TMLR 2024)
  - What it does: feature prediction as a stand-alone video SSL objective.
  - Relation: establishes latent prediction without pixel reconstruction at scale; single-modality video.
  - Verified: arxiv.org/abs/2404.08471; openreview.net/forum?id=QaCCuDfBk2 ("Accepted by TMLR", 2024-08).
- **assran2025vjepa** (V-JEPA 2, arXiv 2025)
  - What it does: scales V-JEPA to more than 1M hours of video; adds action-conditioned planning (V-JEPA 2-AC).
  - Relation: the base model that V-JEPA 2.1 finds weak on dense features, which motivates our dense loss.
  - Verified: arxiv.org/abs/2506.09985; OpenReview CoRR record for the author list.
- **murlabadia2026vjepa** (V-JEPA 2.1, arXiv 2026)
  - What it does: adds a dense predictive loss over masked and context tokens (distance-weighted λ/√d_min) and deep self-supervision at four levels.
  - Relation: the direct source of our 4-depth deep supervision and dense context loss. Its motivation is dense-feature quality, not collapse; it does not discuss position codes.
  - Verified: arxiv.org/abs/2603.14482 and html v3 *(read)*. Title and authors exact: Mur-Labadia, Muckley, Bar, Assran, Sinha, Rabbat, LeCun, Ballas, Bardes.
- **balestriero2025lejepa** (LeJEPA, arXiv 2025)
  - What it does: argues that JEPA embeddings should be isotropic Gaussian, and enforces it with SIGReg (Epps-Pulley test on random 1-D slices); no stop-gradient or teacher.
  - Relation: the source of the SIGReg we tried. It applies SIGReg to per-view global embeddings, and its O(1/N) per-sample gradient bound is consistent with our finding that pooled SIGReg was inert.
  - Verified: arxiv.org/abs/2511.08544; html v3 *(read)*.
- **maes2026leworldmodel** (LeWM, arXiv 2026)
  - What it does: an end-to-end JEPA world model from pixels, with a prediction loss plus SIGReg only.
  - Relation: evidence that SIGReg alone can prevent collapse in a small world model, which contrasts with our dense cross-modal setting.
  - Verified: arxiv.org/abs/2603.19312.
- **wu2026visreg** (VISReg, arXiv 2026)
  - What it does: replaces VICReg covariance with a sliced-Wasserstein sketching term and keeps the variance term.
  - Relation: states that SIGReg suffers "vanishing gradients under collapse" and that covariance is only second-order. Both support our findings on SIGReg and VICReg covariance.
  - Verified: arxiv.org/abs/2606.02572.
- **mo2024connecting** (C-JEPA, NeurIPS 2024)
  - What it does: I-JEPA plus VICReg terms, to fix EMA's failure to prevent full collapse.
  - Relation: the closest precedent for adding VICReg to a JEPA. Our result that the variance hinge on predictions is harmful and covariance favours the position code partly contradicts its recommendation.
  - Verified: arxiv.org/abs/2410.19560; NeurIPS 2024 proceedings PDF *(read)*.
- **bardes2023mcjepa** (MC-JEPA, arXiv 2023)
  - What it does: a JEPA that jointly learns optical flow (motion) and content features in a shared encoder.
  - Relation: an early multi-objective JEPA with VICReg; not cross-modal in our sense.
  - Verified: arxiv.org/abs/2307.12698; OpenReview 9XdLlbxZCC (ICLR 2024, rejected).
- **littwin2024jepa** (NeurIPS 2024)
  - What it does: deep-linear analysis showing JEPA self-distillation is biased toward high-influence (high regression coefficient) features.
  - Relation: theory on which features a JEPA prefers. A perfectly predictable position code is an extreme "high-influence" feature **[our inference]**.
  - Verified: arxiv.org/abs/2407.03475; NeurIPS 2024 proceedings listing.
- **bar2024stochastic** (StoP, ICML 2024)
  - What it does: conditions I-JEPA on stochastic (Gaussian) mask-token positions to model location uncertainty.
  - Relation: documents location-correlated I-JEPA features ("overfit to location features"). A candidate mitigation for our position code.
  - Verified: arxiv.org/abs/2308.00566; PMLR v235 pp. 2944-2958.
- **darcet2025cluster** (CAPI, TMLR 2025)
  - What it does: pure MIM that predicts online-clustered EMA-teacher patch assignments.
  - Relation: **first explicit description of "positional collapse"** and its per-position Sinkhorn fix (see §1). Also reports that iBOT's patch loss without the DINO global loss "results in trivial representations".
  - Verified: arxiv.org/abs/2502.08769; html v3 *(read)*; openreview.net/forum?id=Ycmz7qJxUQ (TMLR).
- **littwin2024enhancing** (EC-IJEPA, NeurIPS 2024 SSL workshop)
  - What it does: conditions I-JEPA's context and target encoders on target and context window positions.
  - Relation: adds positional side information to JEPA encoders. It is the opposite design pressure to our finding that positional information can become the whole representation.
  - Verified: arxiv.org/abs/2410.10773; openreview.net/forum?id=IAqwCSv7kI.
- **mo2024dmtjepa** (DMT-JEPA, arXiv 2024)
  - What it does: builds I-JEPA targets by aggregating semantically similar neighbouring patches, to fix I-JEPA's weak local semantics.
  - Relation: an alternative dense-JEPA target design; no collapse analysis.
  - Verified: arxiv.org/abs/2405.17995.
- **li2025rethinking** (SALT, arXiv 2025)
  - What it does: replaces V-JEPA's EMA teacher with a frozen, pixel-reconstruction-trained teacher.
  - Relation: removes teacher co-adaptation, one route by which target and prediction can co-collapse onto a position code **[our inference]**.
  - Verified: arxiv.org/abs/2509.24317.
- **huang2026text** (TC-JEPA, ICML 2026)
  - What it does: conditions I-JEPA predictions on caption tokens, via sparse cross-attention, to reduce prediction uncertainty.
  - Relation: analogous to our conditioning of the predictor on target modality and sun geometry; reports better training stability.
  - Verified: arxiv.org/abs/2605.03245; OpenReview bmvWC1ainA ("ICML 2026 regular").
- **vanassel2025joint** (NeurIPS 2025)
  - What it does: closed-form theory showing joint embedding needs a weaker alignment condition than reconstruction when irrelevant features have large magnitude.
  - Relation: supports predicting latents rather than pixels when nuisance signal is large, e.g. illumination in lunar rasters **[our inference]**.
  - Verified: arxiv.org/abs/2505.12477; OpenReview UOaLsgn5wb (NeurIPS 2025 spotlight).
- **sobal2022joint** (arXiv 2022, NeurIPS 2022 SSL workshop per arXiv comment)
  - What it does: shows that VICReg or SimCLR JEPAs latch onto fixed "slow" background noise, and gives the trivial solution.
  - Relation: the closest conceptual precedent for a trivially predictable signal that satisfies VICReg (§1).
  - Verified: arxiv.org/abs/2211.10831; PDF *(read)*.
- **wei2024towards** (Latent MIM, ECCV 2024)
  - What it does: dissects latent MIM failures (joint target optimization leading to collapse, MSE, patch redundancy, decoder design).
  - Relation: uses cross-sample pooled cosine similarity as its collapse metric. Its within-image PatchDisc objective could in principle be satisfied by a position code **[our inference]**.
  - Verified: arxiv.org/abs/2407.15837; ECVA ECCV 2024 listing; PDF *(read)*.
- **yamada2026siamjepa** (SiamJEPA, arXiv 2026)
  - What it does: Siamese students plus a Random Shuffle Teacher that removes spatial correspondence from targets.
  - Relation: an explicit anti-positional-shortcut mechanism, plus a position-decodability probe (94.3% without RST); see §1.
  - Verified: arxiv.org/abs/2607.04044; html v3 *(read)*.

### Latent-target, EMA-teacher and MIM SSL

- **baevski2022data2vec** (ICML 2022)
  - What it does: predicts contextualized teacher representations (average of normalized top-K blocks) from a masked view.
  - Relation: precedent for our layer-normed multi-depth EMA targets.
  - Verified: arxiv.org/abs/2202.03555; PDF *(read)*; PMLR v162 pp. 1298-1312.
- **baevski2023efficient** (data2vec 2.0, ICML 2023)
  - What it does: an efficient data2vec (no mask-token encoding, conv decoder, amortized teacher).
  - Relation: an efficiency reference for EMA-target latent prediction.
  - Verified: arxiv.org/abs/2212.07525; PMLR v202 pp. 1416-1429.
- **grill2020bootstrap** (BYOL, NeurIPS 2020)
  - What it does: a predictor plus an EMA target with no negatives.
  - Relation: the origin of the EMA-target plus predictor asymmetry we rely on.
  - Verified: arxiv.org/abs/2006.07733; NeurIPS 2020 proceedings listing.
- **caron2021emerging** (DINO, ICCV 2021)
  - What it does: self-distillation with EMA teacher, centering and sharpening.
  - Relation: centering is per-dimension collapse control. A per-position analogue is a natural fix for a position code **[our inference]**.
  - Verified: arxiv.org/abs/2104.14294; CVF ICCV 2021 listing.
- **oquab2024dinov2** (DINOv2, TMLR 2024)
  - What it does: scaled DINO+iBOT with KoLeo and curated data.
  - Relation: the reference for dense SSL features; its positional artifacts are studied in DVT and "What DINO saw".
  - Verified: arxiv.org/abs/2304.07193; OpenReview a68SUt6zFt (TMLR).
- **simeoni2026dinov3** (DINOv3, TMLR 2026)
  - What it does: a 7B DINO with Gram anchoring against dense-feature degradation in long training.
  - Relation: a documented dense-only failure mode, invisible to global metrics.
  - Verified: arxiv.org/abs/2508.10104; OpenReview 2NlGyqNjns (TMLR, 2026-05-15).
- **zhou2022ibot** (iBOT, ICLR 2022)
  - What it does: masked patch self-distillation with an online tokenizer (EMA teacher).
  - Relation: a patch-level EMA-target loss. CAPI reports that it is unstable without the global DINO term.
  - Verified: arxiv.org/abs/2111.07832; OpenReview ydopy-e6Dg (ICLR 2022 poster).
- **he2022masked** (MAE, CVPR 2022)
  - What it does: pixel reconstruction of heavily masked patches with an asymmetric encoder and decoder.
  - Relation: the pixel-target baseline that latent prediction replaces; "What DINO saw" also finds positional ramps in MAE features.
  - Verified: arxiv.org/abs/2111.06377; CVF CVPR 2022 listing.
- **bao2022beit** (BEiT, ICLR 2022)
  - What it does: masked prediction of discrete dVAE tokens.
  - Relation: an early MIM with tokenized targets.
  - Verified: arxiv.org/abs/2106.08254; OpenReview p-BhZSz59o4 (ICLR 2022 oral).
- **wei2022masked** (MaskFeat, CVPR 2022)
  - What it does: masked prediction of HOG features.
  - Relation: a hand-crafted feature-target MIM, between pixels and latents.
  - Verified: arxiv.org/abs/2112.09133; CVF CVPR 2022 listing.
- **chen2021exploring** (SimSiam, CVPR 2021)
  - What it does: stop-gradient Siamese SSL without EMA or negatives.
  - Relation: introduced the per-channel std of ℓ2-normalized outputs as a collapse monitor. On pooled per-sample outputs it is a cross-sample statistic; computed over tokens, a position code would satisfy it **[our inference]**.
  - Verified: arxiv.org/abs/2011.10566; PDF *(read)*; CVF CVPR 2021.
- **assran2022masked** (MSN, ECCV 2022)
  - What it does: matches masked-view to unmasked-view prototype assignments.
  - Relation: joint-embedding with masking; a precursor to I-JEPA.
  - Verified: arxiv.org/abs/2204.07141; ECVA ECCV 2022 listing.
- **tarvainen2017mean** (Mean Teacher, NIPS 2017)
  - What it does: uses weight-averaged (EMA) teacher targets for consistency.
  - Relation: the origin of the EMA teacher.
  - Verified: arxiv.org/abs/1703.01780; NeurIPS 2017 proceedings listing.
- **he2020momentum** (MoCo, CVPR 2020)
  - What it does: a momentum encoder with a contrastive queue.
  - Relation: the EMA/momentum encoder lineage.
  - Verified: arxiv.org/abs/1911.05722; CVF CVPR 2020 listing.
- **chen2021empirical** (MoCo v3, ICCV 2021)
  - What it does: studies ViT-SSL training instability.
  - Relation: "instability … can be hidden by apparently good results". A precedent for our good-loss, bad-representation failures.
  - Verified: arxiv.org/abs/2104.02057; CVF ICCV 2021 listing.
- **chen2020simple** (SimCLR, ICML 2020)
  - What it does: in-batch contrastive learning.
  - Relation: negatives explicitly penalize sample-independent codes; a reference point for a cross-sample term.
  - Verified: arxiv.org/abs/2002.05709; PMLR v119 pp. 1597-1607.
- **oord2018representation** (CPC, arXiv 2018)
  - What it does: latent prediction with InfoNCE.
  - Relation: cross-sample discrimination in latent prediction; our gap is a non-contrastive, monitor-only cousin.
  - Verified: arxiv.org/abs/1807.03748.
- **wang2021dense** (DenseCL, CVPR 2021)
  - What it does: a dense contrastive loss whose negatives are pooled features of other images.
  - Relation: a dense objective that structurally penalizes position codes **[our inference]**.
  - Verified: arxiv.org/abs/2011.09157; PDF *(read)*; CVF CVPR 2021.
- **park2023self** (ICLR 2023)
  - What it does: compares CL and MIM ViTs. CL attention "collapse[s] into homogeneity"; MIM is more local and texture-oriented.
  - Relation: a different (attention) collapse notion; context for the global-versus-dense trade-off.
  - Verified: arxiv.org/abs/2305.00729; OpenReview azCKuYyS74.

### Collapse: regularizers and theory

- **bardes2022vicreg** (VICReg, ICLR 2022)
  - What it does: variance hinge, invariance and covariance terms.
  - Relation: the terms we use. We find the hinge on predictions harmful and the covariance term favouring the position code.
  - Verified: arxiv.org/abs/2105.04906; OpenReview xm6YD62D1Ub.
- **bardes2022vicregl** (VICRegL, NeurIPS 2022)
  - What it does: VICReg on local features, with location- and feature-based matching plus a global criterion.
  - Relation: keeps a pooled global term that "is sufficient for the vectors to not collapse"; see §3.
  - Verified: arxiv.org/abs/2210.01571; NeurIPS 2022 PDF *(read)*.
- **zbontar2021barlow** (Barlow Twins, ICML 2021)
  - What it does: pushes the cross-correlation matrix toward the identity (redundancy reduction).
  - Relation: a decorrelation alternative to VICReg covariance.
  - Verified: arxiv.org/abs/2103.03230; PMLR v139 pp. 12310-12320.
- **ermolov2021whitening** (W-MSE, ICML 2021)
  - What it does: whitening before an MSE loss.
  - Relation: another second-order collapse control; the same second-order-only critique as VISReg applies.
  - Verified: arxiv.org/abs/2007.06346; PMLR v139 pp. 3015-3024.
- **jing2022understanding** (ICLR 2022)
  - What it does: shows dimensional collapse in contrastive SSL (via augmentation and implicit regularization); proposes DirectCLR.
  - Relation: the canonical reference for our collapse mode 1.
  - Verified: arxiv.org/abs/2110.09348; OpenReview YevsQ05DEN7.
- **hua2021feature** (ICCV 2021)
  - What it does: identifies "dimensional collapse" alongside complete collapse; motivates feature decorrelation.
  - Relation: the co-origin of the dimensional-collapse term.
  - Verified: arxiv.org/abs/2105.00470; CVF ICCV 2021.
- **tian2021understanding** (DirectPred, ICML 2021)
  - What it does: linear-dynamics theory of why BYOL/SimSiam avoid collapse; sets the predictor directly.
  - Relation: the role of predictor, EMA and weight decay in avoiding collapse.
  - Verified: arxiv.org/abs/2102.06810; PMLR v139 pp. 10268-10278.
- **li2022understanding** (ECCV 2022)
  - What it does: partial dimensional collapse in SimSiam; a rank-based (cumulative explained variance) metric.
  - Relation: collapse "not visible in any of the loss metrics", which matches our observation.
  - Verified: arxiv.org/abs/2209.15007; PDF *(read)*; ECVA listing.
- **garrido2023duality** (ICLR 2023)
  - What it does: shows contrastive and non-contrastive SSL are closely related; embedding-dimension and batch effects.
  - Relation: frames why a covariance term alone does not guarantee cross-sample discrimination.
  - Verified: arxiv.org/abs/2206.02574; OpenReview kDEL91Dufpa.
- **halvagal2023implicit** (NeurIPS 2023)
  - What it does: predictor dynamics in non-contrastive SSL give implicit variance regularization; with their IsoLoss, the EMA target can be dropped.
  - Relation: a possible reason our explicit variance hinge is redundant or harmful **[hypothesis]**.
  - Verified: arxiv.org/abs/2212.04858; NeurIPS 2023 listing.
- **shwartzziv2023information** (NeurIPS 2023)
  - What it does: an information-theoretic derivation of VICReg.
  - Relation: theory for what var/cov do and do not guarantee.
  - Verified: arxiv.org/abs/2303.00633; NeurIPS 2023 listing (title differs from arXiv).
- **simon2023stepwise** (ICML 2023)
  - What it does: SSL learns eigenmodes stepwise.
  - Relation: helps read effective-rank trajectories.
  - Verified: arxiv.org/abs/2303.15438; PMLR v202 pp. 31852-31876.
- **balestriero2023cookbook** (arXiv 2023)
  - What it does: a practical SSL guide covering collapse and monitoring.
  - Relation: general reference.
  - Verified: arxiv.org/abs/2304.12210.
- **zhai2023stabilizing** (ICML 2023)
  - What it does: attention entropy collapse goes with transformer training instability; σReparam fix.
  - Relation: a possible mechanism behind grad-norm blow-ups; untested in our setting.
  - Verified: arxiv.org/abs/2303.06296; PMLR v202 pp. 40770-40803.
- **geirhos2020shortcut** (Nat. Mach. Intell. 2020)
  - What it does: the shortcut-learning framework.
  - Relation: frames the position code as a shortcut solution.
  - Verified: arxiv.org/abs/2004.07780; Crossref 10.1038/s42256-020-00257-z (2(11):665-673).

### Collapse and quality monitoring

- **garrido2023rankme** (RankMe, ICML 2023)
  - What it does: effective rank of embeddings as a label-free quality proxy.
  - Relation: our effective-rank monitor. Within-tile rank stays high under a position code.
  - Verified: arxiv.org/abs/2210.02885; PMLR v202 pp. 10929-10974.
- **thilak2024lidar** (LiDAR, ICLR 2024)
  - What it does: the rank of an LDA matrix with clean samples as classes.
  - Relation: the closest existing cross-sample monitor (§2).
  - Verified: arxiv.org/abs/2312.04000; html *(read)*; OpenReview f3g5XpL9Kb (spotlight).
- **agrawal2022alpha** (α-ReQ, NeurIPS 2022)
  - What it does: the power-law decay α of the covariance eigenspectrum as a quality measure.
  - Relation: an alternative spectral monitor, with the same blind spot as RankMe for within-tile position codes **[our inference]**.
  - Verified: proceedings.neurips.cc/paper_files/paper/2022/hash/70596d70542c51c8d9b4e423f4bf2736-Abstract-Conference.html.
- **roy2007effective** (EUSIPCO 2007)
  - What it does: the entropy-based effective rank.
  - Relation: the definition behind RankMe and our rank metric.
  - Verified: zenodo.org/records/40328.
- **wang2020understanding** (ICML 2020)
  - What it does: the alignment and uniformity decomposition.
  - Relation: our cross-sample gap is an alignment-minus-cross-sample-similarity measure in this spirit.
  - Verified: arxiv.org/abs/2005.10242; PMLR v119 pp. 9929-9939.
- **arputharaj2026comparative** (TMLR 2026)
  - What it does: compares label-free representation metrics on 260 models; intrinsic dimension is the most reliable, and all metrics depend on architecture and objective.
  - Relation: evidence that spectral monitors are unreliable proxies.
  - Verified: arxiv.org/abs/2608.23182 (journal-ref TMLR 2026).

### Position information, artifacts and shortcuts

- **darcet2024vision** (Registers, ICLR 2024)
  - What it does: high-norm artifact tokens in background regions, fixed by register tokens.
  - Relation: an artifact, not collapse. A caution when reading token-norm statistics.
  - Verified: arxiv.org/abs/2309.16588; OpenReview 2dnO3LLiJ1 (oral).
- **yang2024denoising** (DVT, ECCV 2024)
  - What it does: grid-like ViT feature artifacts traced "down to the positional embeddings"; a denoiser removes them.
  - Relation: position-driven, input-independent feature components exist even in healthy models.
  - Verified: arxiv.org/abs/2401.02957; ECVA ECCV 2024 listing.
- **wang2024sinder** (ECCV 2024)
  - What it does: repairs DINOv2's singular defects.
  - Relation: artifact literature; low relevance.
  - Verified: arxiv.org/abs/2407.16826; ECVA listing.
- **jiang2025vision** (NeurIPS 2025)
  - What it does: test-time registers without retraining.
  - Relation: artifact literature; low relevance.
  - Verified: arxiv.org/abs/2506.08010; OpenReview bA02DmQN5d.
- **pawlowsky2026dino** (arXiv 2026)
  - What it does: linear probes find input-independent positional ramp channels in SSL ViTs (with learned and RoPE encodings); ALiBi fine-tuning reduces them.
  - Relation: supports that RoPE-based SSL can still encode absolute position (§1).
  - Verified: arxiv.org/abs/2603.16840; html v2 *(read)*.
- **doshi2026biorthogonal** (BFD, arXiv 2026)
  - What it does: ANOVA-style split of tokens into positional and content factors, plus QKᵀ mode analysis.
  - Relation: a principled position-versus-content variance split to pair with our gap.
  - Verified: arxiv.org/abs/2601.05328.
- **chu2023conditional** (CPVT, ICLR 2023)
  - What it does: conv-based conditional positional encoding. Zero padding gives absolute position.
  - Relation: our conv stems can inject absolute position despite 2D RoPE **[our inference]**.
  - Verified: arxiv.org/abs/2102.10882; PDF *(read)*; OpenReview 3KWnuT-R1bh.
- **islam2020much** (ICLR 2020)
  - What it does: CNNs encode a surprising amount of absolute position.
  - Relation: same point as CPVT for the conv stems.
  - Verified: arxiv.org/abs/2001.08248; OpenReview aezVImpzO92.
- **kayhan2020translation** (CVPR 2020)
  - What it does: CNNs exploit absolute location through boundary effects.
  - Relation: same point as CPVT for the conv stems.
  - Verified: arxiv.org/abs/2003.07064; CVF CVPR 2020 listing.
- **doersch2015unsupervised** (ICCV 2015)
  - What it does: context-prediction SSL; found a chromatic-aberration shortcut to absolute position.
  - Relation: the first cross-sample observation of a position shortcut (§1).
  - Verified: arxiv.org/abs/1505.05192; PDF *(read)*; CVF ICCV 2015 listing.
- **zhai2022position** (MP3, ICML 2022)
  - What it does: position prediction as a pretext task.
  - Relation: position treated as a target, not a shortcut; shows ViTs can recover position from content.
  - Verified: arxiv.org/abs/2207.07611; PMLR v162 pp. 26010-26027.
- **wang2023droppos** (DropPos, NeurIPS 2023)
  - What it does: reconstructs dropped positional embeddings, with measures "to avoid trivial solutions".
  - Relation: position-aware SSL with explicit shortcut control.
  - Verified: arxiv.org/abs/2309.03576; NeurIPS 2023 listing.
- **caron2024location** (LOCA, WACV 2024)
  - What it does: patch-level clustering pseudo-labels plus a relative-location prediction task.
  - Relation: location-aware SSL for dense features.
  - Verified: arxiv.org/abs/2212.02400; CVF WACV 2024 listing.
- **ren2023masked** (MJP, CVPR 2023)
  - What it does: positional embeddings "explicitly encode the 2D spatial relationship"; a shuffle-and-occlude PE fix.
  - Relation: positional-embedding leakage as a failure axis.
  - Verified: arxiv.org/abs/2205.12551; CVF CVPR 2023 listing.

### Deep supervision

- **lee2015deeply** (DSN, AISTATS 2015)
  - What it does: auxiliary losses on intermediate layers.
  - Relation: the classic reference behind our 4-depth supervision (with V-JEPA 2.1).
  - Verified: arxiv.org/abs/1409.5185; PMLR v38 pp. 562-570.

### Cross-modal and Earth-observation latent prediction

- **bachmann2022multimae** (MultiMAE, ECCV 2022)
  - What it does: masked autoencoding across co-registered RGB, depth and semantic maps.
  - Relation: the closest pixel-target analogue of our cross-modal masked prediction on co-registered rasters.
  - Verified: arxiv.org/abs/2204.01678; ECVA ECCV 2022 listing.
- **lei2025m3jepa** (M3-JEPA, ICML 2025)
  - What it does: a multimodal JEPA with a multi-gate MoE predictor for cross-modal alignment.
  - Relation: MoE plus JEPA across modalities. Our MoE sits in the encoder, not the predictor.
  - Verified: arxiv.org/abs/2409.05929; PMLR v267 pp. 33902-33917.
- **choudhury2026xjepa** (X-JEPA, WACV 2026)
  - What it does: predicts the target-modality embedding from another modality's context, for cross-modal remote-sensing image retrieval.
  - Relation: cross-modal JEPA in remote sensing. Retrieval-focused, no dense targets.
  - Verified: openaccess.thecvf.com WACV2026 page (bibtex pp. 4355-4364).
- **hossain2026crjepa** (CR-JEPA, arXiv 2026)
  - What it does: modality-specific stems, a shared trunk, same- and cross-modal latent prediction, and SIGReg on retrieval projections.
  - Relation: architecturally the closest to LeLunar (stems, cross-modal JEPA, SIGReg). Retrieval-oriented with global embeddings.
  - Verified: arxiv.org/abs/2606.00706.
- **cornelissen2026mumo** (Le MuMo JEPA, CVPR 2026 Workshops)
  - What it does: a multimodal LeJEPA (RGB with LiDAR or thermal) using fusion tokens, with SIGReg on the pooled joint CLS embedding.
  - Relation: SIGReg applied to pooled embeddings, as in our inert-SIGReg setting.
  - Verified: arxiv.org/abs/2603.24327; CVF CVPR2026_workshops/URVIS page (pp. 8238-8247).
- **astruc2025anysat** (AnySat, CVPR 2025)
  - What it does: a JEPA with scale-adaptive encoders trained on heterogeneous EO data.
  - Relation: the closest EO JEPA foundation model; no position-collapse analysis.
  - Verified: arxiv.org/abs/2412.14123; CVF CVPR 2025 listing.
- **tseng2025galileo** (Galileo, ICML 2025)
  - What it does: a multimodal EO model trained with global (deep-target) and local (shallow-target) masked contrastive losses.
  - Relation: a global-plus-local split with contrastive negatives, which structurally resists sample-independent codes **[our inference]**.
  - Verified: arxiv.org/abs/2502.09356; PMLR v267 pp. 60280-60300.
- **lundqvist2025geojepa** (GeoJEPA, arXiv 2025)
  - What it does: a multimodal JEPA over OpenStreetMap entities and aerial imagery.
  - Relation: geospatial multimodal JEPA; low relevance.
  - Verified: arxiv.org/abs/2503.05774.

---

## Seen but not added to the bib

These papers exist (arXiv or CVF checked) but were judged low relevance:
- HQ-JEPA (arXiv 2605.31068): quantum-hybrid cross-modal RS JEPA with SIGReg.
- Var-JEPA (arXiv 2603.20111; ICML 2026 per arXiv comment).
- UniJEPA (arXiv 2608.07409; ICML 2026 per arXiv comment).
- Gaussian Embeddings (arXiv 2510.05949).
- When Does LeJEPA Learn a World Model? (arXiv 2605.26379).
- Weak-SIGReg (arXiv 2603.05924; ICLR 2026 GRaM workshop per arXiv comment).
- VCReg (arXiv 2306.13292).
- Revealing the Dark Secrets of MIM (CVPR 2023).
- InfoCalib (CVPR 2026 Workshops, multimodal "collapse diagnosis"; not read).
