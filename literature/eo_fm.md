# EO / remote-sensing foundation models and location encoders: verified survey for LeLunar

Compiled 2026-09-30. Every entry below has a matching key in `eo_fm.bib` (85 entries).

**How entries were checked.** Titles and full author lists came from the arXiv API (`export.arxiv.org/api/query`). Venues and pages came from the CVF open-access BibTeX blocks, NeurIPS proceedings `.bib` files, PMLR volume indexes (v202, v235, v267, v306), the ECVA paper list plus Springer DOIs (ECCV 2024), and Crossref DOI records for journals and AAAI. For every design claim marked as read, I pulled the PDF text (pdftotext) and read the relevant method or appendix section myself. I did not rely on summaries for these. OpenReview and dblp returned bot challenges, so the ICLR entries were checked through the arXiv journal-ref, the ICLR virtual site and an OpenReview API search that worked once. See "Unverified items" below.

---

## 1. How prior EO models handle multiple GSDs and resolutions

### 1a. The four strategies, and which papers use each

| Strategy | Mechanism | Papers (bib key) |
|---|---|---|
| **R. Resample to a common grid or canvas** | Every modality is reprojected or resized to one pixel grid (often 10 m) or a fixed image size, and then gets one patch size. | SatMAE `cong2022satmae` (all bands to 96×96 px), CROMA `fuller2023croma` (S1/S2 co-registered, 120×120 px, 8×8 patches), MMEarth `nedungadi2024mmearth` (S2 10 m grid), msGFM `han2024bridging` (resized, 192×192 pretraining), DOFA `xiong2024neural` (224×224), Galileo `tseng2025galileo` (10 m), TerraMind `jakubik2025terramind` (264×264 @ 10 m, DEM from 30 m), SMARTIES `sumbul2025smarties` (resize to input size), Panopticon `waldmann2025panopticon` (random resized crops), AlphaEarth `brown2025alphaearth` (UTM, bilinear to 10 m), OlmoEarth `herzog2026olmoearth` (uniform 10 m), RAMEN `houdre2026ramen` (resampled to a *random target GSD* each iteration), NASA-IBM Lunar FM `fraccaro2026multimodal` (layers snapped to the WAC/NAC anchor pixel grid), LunarFM `gironamata2026lunarfm` (0.5° chips, each modality resized to 112×112). |
| **P. GSD-aware or metric positional encoding** | Position codes or attention biases are scaled by ground distance instead of pixel index. | Scale-MAE `reed2023scale` (sinusoid × g/G), AnySat `astruc2025anysat` (Scale-MAE-style, g = patch size in m, or sub-patch size in m), USat `irvin2023usat` ("superpositional" PE: a coarse token's PE is the mean of the fine-grid PEs it covers), THOR `forgaard2026thor` (2D-ALiBi on metric distance between patch centres), MAESTRO `labatie2026maestro` and RAMEN (Scale-MAE GSD-PE), SenPa-MAE `prexl2024senpa` (per-channel GSD and SRF MLP encodings), Copernicus-FM `wang2025towards` (Fourier "area" metadata = GSD × patch size), GeoRoPE `luo2026georope` (RoPE offsets rescaled by ground distance per token step), Scale-ALiBi `kage2026multi` (ALiBi distance × GSD), MuViT `mantes2026muvit` (microscopy: RoPE on shared world coordinates). |
| **K. Per-resolution or per-modality patch size (pixels per token vary)** | Each band, sensor or GSD gets its own pixels-per-token. The kernel is either learned per modality or resized with FlexiViT PI-resize. | USat (per-band patch projection, per-band patch counts), THOR (per-band projection, random per-GSD patch sizes under a token budget, FlexiViT resize), Copernicus-FM (hypernetwork kernels FlexiViT-reshaped to 16 px for S1/S2, 8 for S3, 4 for S5P, 64 for DEM), MAESTRO (per-modality P_m, token grids may differ), FlexiMo `li2026fleximo` (PI-resize to a GSD-dependent patch size), Any-Optical-Model `li2026any` (kernel bank {16, 32, 64} + PI-resize), OmniSat and AnySat (see G), generic FlexiViT `beyer2023flexivit` and NaViT `dehghani2023patch`. |
| **G. Ground-aligned tokens: one token = one ground cell in every modality** | The tile is cut into P×P-metre cells that are the same for all modalities. Each modality's token for a cell holds however many native pixels fall inside it. | **OmniSat** `astruc2024omnisat` and **AnySat** `astruc2025anysat`. Feature-level (not token-level) analogues: SkySense `guo2024skysense` (per-modality encoders emit maps of equal h×w) and SkySense V2 `zhang2025skysense` (Adaptive Patch Merging shrinks high-resolution tokens stage by stage while MS/SAR keep their resolution). |
| **H. Hypernetwork or dynamic patch embedding** | Patch-embedding weights are generated from sensor metadata. | DOFA (wavelength-conditioned hypernetwork), Copernicus-FM (spectral hypernetwork plus a "variable" hypernetwork driven by LLM embeddings of variable names), FlexiMo (DOFA-style wavelength generator), Panopticon (shared per-channel conv + cross-attention over channels with wavelength/SAR-mode codes; not a hypernetwork, but serves the same purpose), SMARTIES (fixed per-spectral-range projection layers, no hypernetwork), RAMEN (wavelength/polarisation-conditioned channel projector). |

### 1b. Per-model detail, as read in each paper's method section

| Model (key) | Common grid? | Pixels per token vary by modality? | One token = same ground cell across modalities? | Position / scale code | Dynamic patch embedding |
|---|---|---|---|---|---|
| SatMAE `cong2022satmae` | Yes (resize to 96×96, P=8) | No (band groups by native GSD get separate embeddings at the same P) | trivially | sin-cos + group + temporal | No |
| Scale-MAE `reed2023scale` | single modality, rescaled | No | n/a | GSDPE (× g/G) | No |
| Cross-Scale MAE `tang2023cross` | single modality, scale augmentation | No | n/a | cross-scale consistency losses | No |
| SatMAE++ `noman2024rethinking` | single modality, multi-scale | No | n/a | multi-scale reconstruction with conv upsampling | No |
| CROMA `fuller2023croma` | Yes (S1/S2 co-registered) | No (8×8 both) | trivially | 2D-ALiBi / X-ALiBi (grid distance) | No |
| MMEarth `nedungadi2024mmearth` | Yes (S2 10 m grid) | No | trivially | (ConvNeXt V2 FCMAE) | No |
| SkySense `guo2024skysense` | No; separate Swin-H (HR) and ViT-L (S2, S1) encoders | yes, implicitly | feature maps share h×w, fused per location | date PE + geo-context prototypes | No |
| SkySense V2 `zhang2025skysense` | No; one hierarchical backbone | yes, via APM | feature-level alignment | modality prompt tokens | No |
| USat `irvin2023usat` | No; native GSD, S2 20 m bands brought back to 20 m | **Yes** (e.g. 320 m footprint: NAIP 1 m → 16×16 px/token, S2 10 m → 8×8, S2 20 m → 8×8) | **No** (tokens cover 16 m / 80 m / 160 m) | superpositional PE (needs divisible grids) | No |
| msGFM `han2024bridging` | Yes (resized) | No | trivially | — | per-sensor **conv** patch layers |
| DOFA `xiong2024neural` | Yes (224×224), each sensor at its own GSD | No | No | GSD not encoded | Yes (wavelength) |
| **OmniSat** `astruc2024omnisat` | No (native, except SPOT pansharpened and resampled to 1 m) | **Yes** (10 m grid: 0.2 m VHR → 50×50 px/token, S2 → 1 px; 40 m grid: 1 m VHR → 40×40 px, S2 → 4×4 px) | **Yes** | relative Euclidean PE on patch positions | modality-specific CNN (images) / LTAE (time series) |
| **AnySat** `astruc2025anysat` | No (native; some sources upsampled per the paper's Table C) | **Yes**: Δm = P/Rm px; fixed δm sub-patches → modality MLP → shared transformer → 1 vector per cell | **Yes**, with P×P m cells and P re-sampled per batch (e.g. {10, 20, 30} m on TreeSatAI-TS) | GSD-scaled sinusoid: g = P for cells, g = Rm·δm for sub-patches | No |
| Galileo `tseng2025galileo` | Yes (10 m; ERA5/TerraClimate/VIIRS treated as static-in-space) | No (random P ∈ 1..8, same for all; FlexiViT-resizable projection) | trivially | sin-cos space/time, month, channel-group embeddings | No |
| Presto `tseng2023lightweight` | pixel time series only | n/a | n/a | lat/lon as 3-D Cartesian input | No |
| Copernicus-FM `wang2025towards` | No; each modality image comes from the same 0.25° cell at its own pixel size | **Yes** (16 / 8 / 4 / 64 px) | No (modalities pretrained per-modality with MAE + distillation) | Fourier lon/lat, area, time | Yes |
| TerraMind `jakubik2025terramind` | Yes (10 m) | No (16×16) | trivially | geolocation as quarter-degree text tokens | No (FSQ tokenizers) |
| Panopticon `waldmann2025panopticon` | resized crops | No | — | deliberately **no** GSD, time or processing-level information | channel cross-attention |
| SMARTIES `sumbul2025smarties` | Yes (resize) | No | — | — | spectrum-range projections |
| AlphaEarth `brown2025alphaearth` | Yes (10 m) | No | trivially | time codes; decoders use re-gridding and shift-invariant losses for coarser targets | No |
| OlmoEarth `herzog2026olmoearth` | Yes (10 m; NAIP 2.5 m and ERA5 160 m tried, no gain) | No (input resized to mimic patch size) | trivially | sin-cos + temporal + modality embeddings | No |
| THOR `forgaard2026thor` | **No; native 10 m–1000 m** (S1, S2, S3 OLCI/SLSTR) | **Yes** (per-band projection; random per-GSD patch size under a token budget) | No (not enforced) | GSD-aware 2D-ALiBi (metres) | FlexiViT resize |
| RAMEN `houdre2026ramen` | resample to a random target GSD, plus 1×1-conv experts conditioned on log(GSD_m/GSD_target) | No (one token per target pixel) | trivially | Scale-MAE GSD-PE | channel projector |
| MAESTRO `labatie2026maestro` | No | Yes (P_m) | not required | Scale-MAE GSD-PE | No |
| FlexiMo `li2026fleximo` / AOM `li2026any` | one sensor at a time | Yes (PI-resize / kernel bank) | n/a | — | FlexiMo: yes |
| NASA-IBM LFM `fraccaro2026multimodal` | Yes (anchor grid; 256×256 crops) | No (16×16; FlexiViT resize at fine-tuning) | trivially | lat/lon and geometry as binned text tokens | No |
| LunarFM `gironamata2026lunarfm` | Yes (0.5° chips → 112×112; 8×8 patches; footprint changes with latitude) | No | trivially | — | No |
| MOMO `purohit2026momo` | separate per-sensor MAEs at native 0.25–100 m/px, merged by task arithmetic | n/a | n/a | — | — |

**What this means for LeLunar design (a).** "One token = one ground cell for every modality, with per-modality pixels per token and no shared canvas" is **already published**: OmniSat (ECCV 2024) and, more fully, AnySat (CVPR 2025, "spatially consistent patching"). What LeLunar can still claim as different:

- **Tokenizer.** LeLunar uses one fixed-GSD stem per modality whose kernel equals its native pixels per token. OmniSat uses a CNN or LTAE per modality that collapses each patch to a vector. AnySat uses fixed-size sub-patches, a modality MLP, and a shared transformer so that the cell size P can change.
- **No fusion.** LeLunar keeps a separate token grid per modality. Both Astruc et al. models fuse all modalities of a cell into one token with a combiner.
- **Target.** LeLunar predicts other modalities' own token grids. AnySat's JEPA predicts the fused cell embedding.
- **Data regime.** LeLunar covers 60 m–1 km per pixel on the Moon. THOR is the only other native-resolution model spanning about 10 m–1 km, and it does not ground-align tokens.

Coarse products at about one pixel per token or less have a precedent: AnySat treats 250 m MODIS, which is coarser than its 120 m tile, as a single context token with no positional encoding.

---

## 2. JEPA-based EO models: what exactly each one predicts

| Model (key, venue) | Context → target | Target representation | Target network | Predictor and its conditioning | Loss / anti-collapse |
|---|---|---|---|---|---|
| SAR-JEPA `li2024predicting` (ISPRS JPRS 2024) | SAR only; local masked patches | **Hand-crafted multi-scale SAR gradient features** of the unseen context | none (fixed feature extractor) | ViT | feature prediction; the target is domain-engineered, not learned |
| **AnySat** `astruc2025anysat` (CVPR 2025) | Student sees the tile with patches dropped (5 rectangles), plus random modality masking and 50 % timestamp masking | **Teacher's fused multimodal cell embedding** (after the modality combiner) at the dropped cells | EMA teacher (0.996) on the full, unmasked input | self-attention over surviving cell tokens plus a learned `f_drop` + PE. Not conditioned on a target modality, since the target is fused. Depth is given as 3 blocks in the main text and 1 block in the appendix. | L2, plus a cross-modal patch InfoNCE on per-modality cell embeddings |
| **Galileo** `tseng2025galileo` (ICML 2025) | Online encoder sees some channel groups. **Global task: the target view uses channel groups disjoint from the context view** (i.e. cross-modal), with space or time structured masking. | Global: EMA target-encoder tokens at a **modality-specific exit depth** (S1/S2 full depth, other groups half, pseudo-label maps 0). Local: the EMA encoder's **linear projection** only. | EMA (0.996 → 1) | **Queries = target tokens' position + time + month + channel-group (modality) embeddings, cross-attending to the visible encodings.** This is the same mechanism as LeLunar's target-modality-conditioned cross-attention predictor. | PatchDisc (InfoNCE; batch negatives for global), not L2 |
| **OlmoEarth** `herzog2026olmoearth` (CVPR 2026) | Modality-aware masking: each bandset is encode-only, decode-only, both, or unused; maps are decode-only, so it reconstructs missing bandsets from other bandsets | **Frozen, randomly initialised linear projection** of each modality's patches ("Latent MIM Lite") | none (fixed random projection) | Decoder with mask tokens + position / time / **modality** embeddings cross-attending to encoder tokens (depth 4) | Modality patch discrimination (negatives only from the same modality) + instance contrastive (×0.1). **Reports that full EMA Latent MIM collapsed.** |
| REJEPA `choudhury2025rejepa` (CVPRW 2025) | Unimodal; context tokens → target tokens | abstract target-token representations | (not examined) | — | VICReg against collapse |
| **X-JEPA** `choudhury2026x` (WACV 2026) | **Context from one modality → embedding of the target modality** (S1↔S2, RGB↔Sentinel) | semantic embedding of the target modality | (not examined) | — | "Prediction Space Alignment" loss |
| CR-JEPA `hossain2026cr` (arXiv 2026) | Modality-specific stems + shared trunk; masked latent targets **within and across** modalities | masked latent target features | (not examined) | — | SIGReg (LeJEPA-inspired) on retrieval projections |
| HQ-JEPA `hossain2026hq` (arXiv 2026) | Paired S1/S2; visible → masked | masked target representations | (not examined) | — | latent prediction + cross-modal alignment + SIGReg + a quantum-fidelity loss |
| UA-JEPA `junghare2026ua` (PRL 2026) | retrieval (title-level only) | — | — | — | uncertainty-aware (not read) |
| SpectralEarth-FM `aitalibraham2026spectralearth` (arXiv 2026) | global multi-sensor view ↔ single-sensor local views (HSI + MSI + LST + SAR) | "JEPA-style" matching of representations | (not examined) | — | — |
| GeoMeld-FM `hasan2026geomeld` (arXiv; CVPRW 2026 per comment) | multi-pretext MAE + JEPA + caption contrastive | — | — | — | — |
| AlignJEPA `hossain2026alignjepa` (arXiv 2026) | masked AnySat visual tokens → **RemoteCLIP text embeddings** | text embedding | frozen text encoder | multi-scale predictive aligner | prediction + bidirectional contrastive |
| Sat-JEPA-Diff `komurcu2026sat` (ICLR 2026 WS) | I-JEPA predicts future semantic representations that steer a frozen Stable Diffusion model (S2 forecasting) | — | — | — | — |
| T-SAR-JEPA `woldesenbet2026t` (arXiv; IGARSS 2026 per comment) | SAR-JEPA encoder + temporal transformer forecasting future latent states | future latents | — | — | anomaly detection |
| GeoJEPA `lundqvist2025geojepa` (MSc thesis) | multimodal JEPA over OSM attributes, geometries and aerial images | — | — | — | — |
| JEDI `bhaumik2026jedi`, LeWM-EO `albughdadi2026from` (arXiv 2026) | distillation from an I-JEPA teacher / a JEPA world model for cloud-observability forecasting | — | — | — | — |

**Nearby models that predict tokens or latents but are not JEPAs:**
- TerraMind and the NASA-IBM Lunar FM do **cross-modal prediction of discrete FSQ tokens** from frozen per-modality tokenizers, with cross-entropy.
- msGFM does **cross-sensor pixel reconstruction**, e.g. masked DSM → RGB pixels.
- AlphaEarth reconstructs every source from its embedding with **conditional implicit decoders**, plus teacher–student consistency and batch uniformity on S^63.
- Copernicus-FM and DOFA use MAE plus distillation from DINOv2 or SoftCon.
- Panopticon uses DINOv2 self-distillation with cross-sensor views as augmentations.

**Where LeLunar sits.** Its source-masked encoder, predictor conditioned on target modality and target sun geometry, and EMA targets as per-modality token grids of other modalities are closest to three works:
- **Galileo's global task.** It has the same target-query cross-attention with disjoint channel groups and EMA targets, but it resamples everything to 10 m and uses a contrastive loss.
- **X-JEPA and CR-JEPA.** Both do cross-modal latent prediction, but for retrieval and at the image level.
- **AnySat.** It is multimodal and multi-resolution, but its targets are fused and its predictor ignores the target modality.

None of these condition the predictor on illumination or acquisition geometry. OlmoEarth's report that EMA latent MIM collapses on EO data matches the collapse issues LeLunar has run into, and is worth citing.

---

## 3. Acquisition and illumination-geometry conditioning (design c)

- **AlphaEarth** `brown2025alphaearth`: each source decoder is conditioned on a time code and on "metadata that is only relevant to the act of measurement": S1 platform heading and orbital inclination, ALOS-2 PALSAR-2 pass direction and antenna pointing. This is the closest EO analogue to conditioning the predictor on target geometry.
- **NASA-IBM Lunar FM** `fraccaro2026multimodal`: per-tile solar incidence, emission, phase and azimuth angles, sub-solar and tile-centre coordinates, and GSD are given to the **encoder** as binned text tokens. The paper reports "illumination-consistent reflectance from geometry" in generation. **This is the most direct lunar precedent for (c).**
- **SARFormer** `prexl2025sarformer`: an acquisition-parameter encoding module for SAR geometry.
- **SenPa-MAE** `prexl2024senpa`: per-channel sensor-parameter encodings (spectral response function and GSD).
- **Copernicus-FM** `wang2025towards`: Fourier metadata for location, area and time.
- **Prithvi-EO-2.0** `szwarcman2026prithvi`: lat/lon and date embeddings added as a weighted bias, with random dropping.
- **S-NeRF** `derksen2021shadow` and **Sat-NeRF** `mari2022sat`: explicit sun-direction inputs to model shadows in satellite NeRFs. These are not foundation models.
- The opposite design choice, **invariance**, appears in AnySat (JEPA chosen because EO pixels vary with "weather, time of day, or acquisition angle") and Panopticon (no GSD or time information, cross-sensor views treated as augmentations).

## 4. Location encoders and Slepian functions (design b)

- **Families.**
  - Wrap / sinusoidal: Mac Aodha `macaodha2019presence`, SINR `cole2023spatial`.
  - Multi-scale grid cells: Space2Vec `mai2020multi`, Sphere2Vec `mai2023sphere2vec`, CSP `mai2023csp`.
  - Hierarchical random Fourier features on an equal-earth projection: GeoCLIP `vivancocepeda2023geoclip`.
  - Spherical harmonics with a SIREN network: `russwurm2024geographic`, used by SatCLIP `klemmer2025satclip` (L = 10 / 40).
  - Retrieval-augmented: RANGE `dhakal2025range`.
  - Distillation: SLED `lane2026sled`.
  - Localised spherical wavelets: `cai2025no`.
  - Survey and benchmark: `mai2022review`, TorchSpatial `wu2024torchspatial`.
- **In EO foundation models:**
  - Prithvi-EO-2.0: 1-D sin-cos of lat and lon.
  - Copernicus-FM: Fourier lon/lat.
  - Presto: 3-D Cartesian coordinates.
  - Galileo: lat/lon as a static input.
  - TerraMind: quarter-degree text tokens.
  - NASA-IBM LFM: binned text tokens.
  - SkySense: 4096 region prototypes.

  **None uses Slepian functions.**
- **Slepian functions in ML.** The only location-encoding use I found is **Rao et al., ICML 2026** `rao2026localized` (PMLR 306:103850–103873; arXiv 2602.00392). It builds spherical-cap Slepians of bandlimit L_r, keeps the top N_Θ = ⌈(1−cosΘ)/2·(L_r+1)²⌉ well-concentrated modes (the Shannon number), computes them once and rotates them to the cap centre, and adds a hybrid variant that concatenates regional Slepians with a coarse global spherical-harmonic basis at L_g ≪ L_r. They argue it is pole-safe and preserves spherical distance, and say explicitly that earlier Slepian work analysed geophysical signals rather than encoding positions. Code is `arjunarao619/SlepianPosEnc`, tag `v1.0`, commit 123ca76.

  Search queries were: arXiv `abs:Slepian AND (encoding|encoder|location|positional)`, arXiv `abs:Slepian AND (neural network|deep learning|machine learning|transformer)`, and two web searches. Other ML uses of Slepians that turned up are graph Slepians (SlepNet) and Slepian wavelets, neither of which is a location encoder. I found no EO or planetary foundation model that uses Slepian location encoding.
- **Theory to cite:**
  - Simons, Dahlen & Wieczorek 2006 `simons2006spatiospectral`.
  - Simons & Dahlen 2006 on polar caps and the polar gap `simons2006spherical`.
  - Wieczorek & Simons 2005 on localised spectral analysis `wieczorek2005localized`.
  - Bates et al. 2017 on efficient Slepian computation `bates2017efficient`, the cap reference Rao et al. cite.
  - SHTOOLS `wieczorek2018shtools`.

## 5. Conv stems vs linear patchify in EO (design e)

- **OmniSat, appendix Table A-1.** Embedding 50×50-px VHR patches with a linear layer does worse than the CNN encoder. The authors write that "50×50 patches are too large to use linear projection". This is the most direct EO evidence for (e).
- **msGFM** uses per-sensor convolution layers before patchifying.
- **MMEarth** is fully convolutional (ConvNeXt V2 FCMAE).
- **AlphaEarth**'s STP encoder has a 3×3-conv "precision" pathway.
- **RAMEN** uses 1×1-conv experts after resampling.
- **AnySat** uses an MLP per sub-patch plus a transformer, with no conv.
- **TerraMind** and **Galileo** use linear projections.
- The generic CV reference is Xiao et al. `xiao2021early`, but other surveys may cover it.

---

## 6. Unverified items and caveats

1. **Venues I could not find (cite as arXiv).**
   - DOFA `xiong2024neural`: no peer-reviewed version turned up in Crossref, and the v3 of 2025-10 adds "DOFA+".
   - USat, Presto, Prithvi (2023), AlphaEarth, THOR, GeoRoPE, CR-JEPA, HQ-JEPA, AlignJEPA, SpectralEarth-FM, JEDI, LeWM-EO, SLED, `cai2025no`, NASA-IBM LFM and LunarFM: arXiv only as of 2026-09-30.
2. **Venues taken only from arXiv comments.**
   - GeoMeld (CVPR Workshop 2026).
   - T-SAR-JEPA (to appear, IGARSS 2026).
   - Sat-JEPA-Diff (ICLR 2026 ML4RS workshop).
   - Rao et al., ICLR 2026 `rao2026measuring`: arXiv comment plus the ICML paper's bibliography, because OpenReview was blocked.
   - Space2Vec ICLR 2020: arXiv journal-ref plus the iclr.cc virtual poster page from search, not opened directly.
3. **UA-JEPA** `junghare2026ua`: only Crossref metadata; I did not read the content.
4. **AnySat inconsistencies.** Predictor depth is "3 self attention blocks" in the main text and "a single self-attention block" in appendix C. On sub-patch sizes, the main text says "1 pixel for very high-resolution images and 10 pixels for time series", while the appendix says "4×4 pixels for PASTIS and 10×10 pixels for FLAIR". These look contradictory, so I did not quote specific δm values.
5. **OlmoEarth on CVF** uses the model's earlier name, "Helios", in its abstract. The title and authors match arXiv 2511.13655.
6. **NASA-IBM LFM.** How the 500 m UV layer is gridded within 100 m WAC tiles is not fully clear from the text: layers are "snapped to the anchor pixel grid" with per-channel resolution. **Himanshu Patil is a co-author**, so please check that description against what the team knows.
7. **Author-list and title variants.**
   - SAR-JEPA's second author is "Wei Yang" in the journal record (used here) and "Yang Wei" on arXiv.
   - SatMAE++: CVF gives "Rao Muhammad Anwer" (used), arXiv gives "Anwar".
   - FlexiMo's TGRS version has 6 authors; arXiv v1 has 4.
   - Sphere2Vec's journal version has 8 authors and a different subtitle than arXiv v1 (5 authors).
   - PANGAEA's journal title differs from arXiv.
   - MuViT's first author is written "Mantes, Albert Dominguez" in CVF BibTeX (used as-is), so the surname may be "Dominguez Mantes".
8. **SlepianPosEnc repository.** The README BibTeX lists `author={Anonymous}`; use the PMLR author list. The README capitalises "High-Resolution" while PMLR and arXiv use "High-resolution" (used). The GitHub API reports **no license**, which matters if LeLunar vendors the code. Tag `v1.0` is confirmed.
9. **Numbers deliberately left out.** I did not transcribe any benchmark numbers. Dataset scales in the one-liners are quoted only where I read them.

---

## 7. Per-entry notes

Format: **key**: what it does. *Relation to LeLunar.* Verification URL.

### Core EO foundation models
- **cong2022satmae**: MAE for temporal and multispectral imagery; spectral band groups with separate patch embeddings; temporal encoding. *Resample-to-canvas baseline; groups bands by GSD but resizes them all to 96×96.* https://proceedings.neurips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html ; arXiv 2207.08051
- **reed2023scale**: GSD-scaled positional encoding and a Laplacian-pyramid decoder for multiscale MAE. *The canonical GSD-PE; LeLunar's fixed-GSD stems avoid needing it by fixing ground cells.* https://openaccess.thecvf.com/content/ICCV2023/html/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.html
- **tang2023cross**: scale augmentation plus contrastive and generative cross-scale consistency on MAE. *Single-modality multi-scale SSL.* https://proceedings.neurips.cc/paper_files/paper/2023/hash/3fadcbd0437f4717723ff3f6f7216800-Abstract-Conference.html
- **noman2024rethinking**: SatMAE++, multi-scale pretraining with conv upsampling reconstruction. *Single modality; pixel targets.* https://openaccess.thecvf.com/content/CVPR2024/html/Noman_Rethinking_Transformers_Pre-training_for_Multi-Spectral_Satellite_Imagery_CVPR_2024_paper.html
- **mendieta2023towards**: GFM, continual pretraining from an ImageNet-22k teacher on GeoPile. *Background only.* https://openaccess.thecvf.com/content/ICCV2023/html/Mendieta_Towards_Geospatial_Foundation_Models_via_Continual_Pretraining_ICCV_2023_paper.html
- **fuller2023croma**: radar–optical contrastive plus a joint MAE decoder, with X-ALiBi and 2D-ALiBi. *Co-registered 10 m inputs; ALiBi is the root of THOR's metric ALiBi.* https://proceedings.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html
- **wang2024decoupling**: DeCUR, decouples common and unique multimodal representations via redundancy reduction (radar–optical, RGB–elevation, RGB–depth). *A multimodal SSL objective that is not predictive.* https://www.ecva.net/papers.php ; https://doi.org/10.1007/978-3-031-73397-0_17
- **wang2023ssl4eo**: SSL4EO-S12, a global multi-seasonal S1/S2 SSL dataset with MoCo, DINO, MAE and data2vec baselines. *Dataset background.* https://doi.org/10.1109/MGRS.2023.3281651 ; arXiv 2211.07044
- **bastani2023satlaspretrain**: large S2 and NAIP labelled pretraining dataset (302M labels, 137 categories). *Background.* https://openaccess.thecvf.com/content/ICCV2023/html/Bastani_SatlasPretrain_A_Large-Scale_Dataset_for_Remote_Sensing_Image_Understanding_ICCV_2023_paper.html
- **jakubik2023foundation**: Prithvi, a ViT MAE on HLS time series. *NASA/IBM EO lineage of the lunar FM.* https://arxiv.org/abs/2310.18660
- **szwarcman2026prithvi**: Prithvi-EO-2.0, multitemporal HLS FM with lat/lon and date embeddings added as weighted biases, with random dropping. *Example of a location/time encoding inside an EO FM.* https://doi.org/10.1109/TGRS.2025.3642610 ; arXiv 2412.02732
- **hong2024spectralgpt**: 3-D spatial-spectral tokens and multi-target reconstruction for spectral imagery. *Spectral tokenisation, single sensor family.* https://doi.org/10.1109/TPAMI.2024.3362475
- **guo2024skysense**: factorised multimodal spatiotemporal encoder (HR optical, S2, S1) with multi-granularity contrastive learning and geo-context prototypes. *Aligns modalities at the feature-map level, not the token level; location enters as region prototypes.* https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.html
- **zhang2025skysense**: SkySense V2, one hierarchical backbone for all modalities; Adaptive Patch Merging downsamples high-resolution tokens while MS/SAR keep their resolution; modality prompts and MoE. *A feature-level way to harmonise GSD, in contrast with LeLunar's input-level stems.* https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_SkySense_V2_A_Unified_Foundation_Model_for_Multi-modal_Remote_Sensing_ICCV_2025_paper.html
- **irvin2023usat**: per-band patch projection at native GSD, more tokens for finer bands, spectral-group pooling and superpositional PE (NAIP + S2). *Closest earlier native-resolution tokenizer, but tokens cover different ground areas per band.* https://arxiv.org/abs/2312.02199
- **han2024bridging**: msGFM, per-sensor conv patch embeddings, shared Swin, SimMIM with cross-sensor reconstruction (e.g. DSM → RGB), MoE. *Cross-modal prediction in pixel space; conv stems.* https://openaccess.thecvf.com/content/CVPR2024/html/Han_Bridging_Remote_Sensors_with_Multisensor_Geospatial_Foundation_Models_CVPR_2024_paper.html
- **xiong2024neural**: DOFA, wavelength-conditioned hypernetwork patch embedding with continual MIM plus distillation across five sensors. *Hypernetwork strategy; does not encode GSD.* https://arxiv.org/abs/2403.15356
- **astruc2024omnisat**: tile split into ground-consistent patches across VHR, S2 and S1; modality-specific CNN / LTAE encoders; combiner; per-patch cross-modal contrastive plus MAE. *Prior art for one token = one ground cell (design a) and evidence for conv stems (design e).* https://www.ecva.net/papers.php ; https://doi.org/10.1007/978-3-031-73390-1_24
- **astruc2025anysat**: JEPA with a scale-adaptive patch encoder over P-metre cells shared by all modalities, GSD-scaled PE, and training on 5 datasets with 11 sensors (0.2–250 m). *The closest prior art to LeLunar overall: ground-aligned multi-resolution tokens plus a JEPA. It differs in its fused targets, target-agnostic predictor and lack of geometry conditioning.* https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html
- **tseng2025galileo**: global and local latent prediction with EMA targets at modality-dependent depth; cross-attention predictor queried by target position/time/month/channel-group embeddings; PatchDisc loss. *Closest prior art for cross-modal latent prediction with a target-conditioned predictor (design d); everything resampled to 10 m.* https://proceedings.mlr.press/v267/tseng25a.html
- **tseng2023lightweight**: Presto, a pixel-time-series transformer with lat/lon as 3-D Cartesian input. *Location as an input feature.* https://arxiv.org/abs/2304.14065
- **nedungadi2024mmearth**: 12 modalities (pixel-level and image-level) on the S2 10 m grid; multi-pretext ConvNeXt V2 MAE. *Resample baseline; conv architecture.* https://www.ecva.net/papers.php ; https://doi.org/10.1007/978-3-031-73039-9_10
- **wang2025towards**: Copernicus-FM, spectral and LLM-variable hypernetworks with FlexiViT-reshaped per-modality patch sizes (S1–S5P, DEM), Fourier metadata (lon/lat, area, time), MIM plus distillation. *Per-modality patch size and metadata conditioning, but no ground-aligned tokens and no cross-modal targets.* https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Towards_a_Unified_Copernicus_Foundation_Model_for_Earth_Vision_ICCV_2025_paper.html
- **jakubik2025terramind**: any-to-any generative EO model; FSQ tokenizers per modality; dual-scale early fusion; discrete-token targets. *Architecture the NASA-IBM Lunar FM adapts; common 10 m grid; discrete, not continuous, targets.* https://openaccess.thecvf.com/content/ICCV2025/html/Jakubik_TerraMind_Large-Scale_Generative_Multimodality_for_Earth_Observation_ICCV_2025_paper.html
- **waldmann2025panopticon**: DINOv2 with cross-sensor views as augmentations, channel subsampling and channel cross-attention patch embedding. *Seeks invariance to GSD and sensor instead of explicit handling.* https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Waldmann_Panopticon_Advancing_Any-Sensor_Foundation_Models_for_Earth_Observation_CVPRW_2025_paper.html
- **sumbul2025smarties**: spectrum-aware projection layers per wavelength range, cross-sensor token mixup, MAE. *Resize to common input; sensor-agnostic.* https://openaccess.thecvf.com/content/ICCV2025/html/Sumbul_SMARTIES_Spectrum-Aware_Multi-Sensor_Auto-Encoder_for_Remote_Sensing_Images_ICCV_2025_paper.html
- **brown2025alphaearth**: STP video encoder producing a 10 m embedding field; conditional implicit decoders per source (time code plus measurement metadata); teacher–student and text objectives. *Acquisition-metadata conditioning of decoders is the closest analogue to design (c); inputs resampled to 10 m.* https://arxiv.org/abs/2507.22291
- **herzog2026olmoearth**: Latent MIM Lite (frozen random-projection targets), modality-aware masking, modality patch discrimination. *Cross-modal latent prediction without an EMA target, motivated by collapse.* https://openaccess.thecvf.com/content/CVPR2026/html/Herzog_OlmoEarth_Stable_Latent_Image_Modeling_for_Multimodal_Earth_Observation_CVPR_2026_paper.html
- **forgaard2026thor**: native-resolution S1/S2/S3 (10–1000 m), per-band projection, randomised per-GSD patch sizes, GSD-aware ALiBi, MAE with map-prediction heads. *Closest in GSD span to LeLunar; ground-cell alignment not enforced.* https://arxiv.org/abs/2601.16011
- **houdre2026ramen**: resamples each modality to a user or random target GSD with ratio-conditioned 1×1-conv experts and reconstructs at native resolution. *Output resolution becomes a control parameter; this is the resampling alternative to (a).* https://openaccess.thecvf.com/content/CVPR2026/html/Houdre_RAMEN_Resolution-Adjustable_Multimodal_Encoder_for_Earth_Observation_CVPR_2026_paper.html
- **labatie2026maestro**: MAE with per-modality patch sizes, GSD-PE, token-based fusion strategies and spectral-prior target normalisation. *Per-modality patch sizes.* https://openaccess.thecvf.com/content/WACV2026/html/Labatie_MAESTRO_Masked_AutoEncoders_for_Multimodal_Multitemporal_and_Multispectral_Earth_Observation_WACV_2026_paper.html
- **bountos2025fomo**: FoMo-Bench and FoMo-Net for forest monitoring; flexible multi-modal, multi-scale inputs. *Background.* https://doi.org/10.1609/aaai.v39i27.35002
- **prexl2024senpa**: per-channel sensor-parameter (spectral response and GSD) encodings via MLPs in an MAE. *GSD and sensor metadata conditioning.* https://doi.org/10.1007/978-3-031-85187-2_20 ; arXiv 2408.11000
- **prexl2025sarformer**: ViT with an acquisition-parameter encoding module for one or several SAR images. *Geometry conditioning for SAR (design c).* https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Prexl_SARFormer_-_An_Acquisition_Parameter_Aware_Vision_Transformer_for_Synthetic_CVPRW_2025_paper.html

### Resolution / patch-size handling
- **beyer2023flexivit**: random patch sizes during training plus pseudo-inverse resize of the patch-embedding weights. *Mechanism behind Galileo, THOR, Copernicus-FM, FlexiMo and the NASA-IBM LFM; LeLunar instead fixes the kernel per GSD.* https://openaccess.thecvf.com/content/CVPR2023/html/Beyer_FlexiViT_One_Model_for_All_Patch_Sizes_CVPR_2023_paper.html
- **dehghani2023patch**: NaViT, native-resolution and aspect-ratio ViT through sequence packing. *Generic native-resolution tokens.* https://proceedings.neurips.cc/paper_files/paper/2023/hash/06ea400b9b7cfce6428ec27a371632eb-Abstract-Conference.html
- **li2026fleximo**: wavelength hypernetwork plus parameter-free PI-resize to a GSD-dependent patch size. *Per-GSD patch size by kernel resizing.* https://doi.org/10.1109/TGRS.2026.3656362
- **li2026any**: Any-Optical-Model, channel-wise tokens, a bank of multi-scale patch kernels with PI-resize, multi-scale semantic alignment. *Per-resolution kernels.* https://doi.org/10.1609/aaai.v40i8.37583
- **luo2026georope**: rescales RoPE offsets by ground distance per token step, as an adapter for pretrained RS models. *Metric positional encoding.* https://arxiv.org/abs/2606.14760
- **kage2026multi**: Scale-ALiBi, ALiBi bias scaled by GSD across high- and low-resolution optical and SAR. *Metric attention bias.* https://arxiv.org/abs/2604.10347
- **mantes2026muvit**: multi-resolution views of one image embedded in shared world coordinates with RoPE (microscopy). *Non-EO analogue of multi-GSD tokens in one encoder.* https://openaccess.thecvf.com/content/CVPR2026/html/Mantes_MuViT_Multi-Resolution_Vision_Transformers_for_Learning_Across_Scales_in_Microscopy_CVPR_2026_paper.html
- **xiao2021early**: early convolutional stems improve ViT optimisation. *Generic reference for design (e).* https://proceedings.neurips.cc/paper_files/paper/2021/hash/ff1418e8cc993fe8abcfe3ce2003e5c5-Abstract.html

### JEPA / latent prediction for RS
- **li2024predicting**: SAR-JEPA, predicts multi-scale SAR gradient features of masked local regions. *First JEPA-style EO work; hand-crafted targets.* https://doi.org/10.1016/j.isprsjprs.2024.09.013
- **choudhury2025rejepa**: unimodal JEPA with VICReg for retrieval. *JEPA in EO, unimodal.* https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Choudhury_REJEPA_A_Novel_Joint-Embedding_Predictive_Architecture_for_Efficient_Remote_Sensing_CVPRW_2025_paper.html
- **choudhury2026x**: X-JEPA, predicts the target modality's embedding from another modality, with a PSA loss. *Explicitly cross-modal latent prediction (design d), for retrieval.* https://openaccess.thecvf.com/content/WACV2026/html/Choudhury_X-JEPA_A_Novel_Joint_Learning_Cross-Modal_Predictive_Alignment_Framework_for_WACV_2026_paper.html
- **hossain2026cr**: CR-JEPA, modality-specific stems, shared trunk, within- and cross-modal masked latent prediction, SIGReg. *Cross-modal JEPA with per-modality stems.* https://arxiv.org/abs/2606.00706
- **hossain2026hq**: HQ-JEPA, S1/S2 latent prediction plus alignment, SIGReg and a quantum-fidelity loss. *Cross-modal JEPA.* https://arxiv.org/abs/2605.31068
- **hossain2026alignjepa**: predicts text embeddings from masked AnySat tokens. *Uses AnySat as a backbone.* https://arxiv.org/abs/2608.15456
- **junghare2026ua**: uncertainty-aware JEPA for retrieval (metadata only). https://doi.org/10.1016/j.patrec.2026.08.007
- **aitalibraham2026spectralearth**: hyperspectral plus multisensor hierarchical FM trained with a JEPA-style global/local objective. *Multisensor JEPA-style pretraining.* https://arxiv.org/abs/2605.21075
- **hasan2026geomeld**: GeoMeld dataset and GeoMeld-FM (MAE + JEPA + caption contrastive). https://arxiv.org/abs/2604.10591
- **komurcu2026sat**: I-JEPA features steer a frozen Stable Diffusion model for S2 forecasting. https://arxiv.org/abs/2603.13943
- **woldesenbet2026t**: forecasts SAR latent states for anomaly detection. https://arxiv.org/abs/2606.05700
- **lundqvist2025geojepa**: multimodal JEPA on OSM and aerial imagery (thesis). https://arxiv.org/abs/2503.05774
- **bhaumik2026jedi**: distils an I-JEPA teacher into SegFormer for cropland segmentation. https://arxiv.org/abs/2609.07915
- **albughdadi2026from**: JEPA world model (LeWorldModel) for cloud-observability forecasting. https://arxiv.org/abs/2607.13651

### Planetary foundation models
- **fraccaro2026multimodal**: NASA-IBM Lunar FM, TerraMind-style masked discrete-token model on SomBench (WAC 100 m and NAC 1 m families); acquisition geometry as encoder tokens; FlexiViT. *The most direct precedent. LeLunar differs in native-resolution stems, continuous JEPA targets and geometry conditioning of the predictor.* https://arxiv.org/abs/2609.13283
- **gironamata2026lunarfm**: LunarFM, multimodal MAE on 18 co-registered channels from six instruments (±70°), 0.5° chips; names JEPA as future work. *Lunar resample-to-grid baseline.* https://arxiv.org/abs/2607.22408
- **sander2026moon**: unified any-to-any transformer across lunar images, DEMs, normals and albedo (shape and albedo from shading). *Lunar cross-modal translation.* https://doi.org/10.1016/j.isprsjprs.2026.04.008
- **purohit2026momo**: Mars multi-sensor FM (HiRISE, CTX, THEMIS; 0.25–100 m/px) built by merging per-sensor MAEs with Equal-Validation-Loss checkpoint selection. *A model-merging alternative to joint multi-GSD tokenisation.* https://openaccess.thecvf.com/content/CVPR2026/html/Purohit_MOMO_Mars_Orbital_MOdel_Foundation_Model_for_Mars_Orbital_Applications_CVPR_2026_paper.html

### Geometry conditioning
- **derksen2021shadow**: S-NeRF, sun-direction-conditioned shadow modelling. https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Derksen_Shadow_Neural_Radiance_Fields_for_Multi-View_Satellite_Photogrammetry_CVPRW_2021_paper.html
- **mari2022sat**: Sat-NeRF, adds transient objects and shadows with RPC cameras. https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Mari_Sat-NeRF_Learning_Multi-View_Satellite_Photogrammetry_With_Transient_Objects_and_Shadow_CVPRW_2022_paper.html

### Benchmarks and position papers
- **lacoste2023geo**: GEO-Bench, 6 classification and 6 segmentation tasks. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a0644215d9cff6646fa334dfa5d29c5a-Abstract-Datasets_and_Benchmarks.html
- **marsocci2026pangaea**: PANGAEA benchmark across resolutions, sensors and temporalities. https://doi.org/10.1109/MGRS.2025.3628194 ; arXiv 2412.04204
- **rolf2024position**: position paper arguing satellite data is a distinct ML modality. *Motivation for modality-specific design.* https://proceedings.mlr.press/v235/rolf24a.html

### Location encoders
- **macaodha2019presence**: wrap-encoded lat/lon geographic prior. https://openaccess.thecvf.com/content_ICCV_2019/html/Aodha_Presence-Only_Geographical_Priors_for_Fine-Grained_Image_Classification_ICCV_2019_paper.html
- **mai2020multi**: Space2Vec, multi-scale grid-cell sinusoids. https://iclr.cc/virtual/2020/poster/1778 ; arXiv 2003.00824
- **mai2022review**: review of location encoding for GeoAI. https://doi.org/10.1080/13658816.2021.2004602
- **mai2023sphere2vec**: multi-scale spherical encodings that preserve spherical distance. https://doi.org/10.1016/j.isprsjprs.2023.06.016
- **mai2023csp**: contrastive spatial pretraining pairing location encoders with images. https://proceedings.mlr.press/v202/mai23a.html
- **cole2023spatial**: SINR, implicit neural species range maps from coordinates. https://proceedings.mlr.press/v202/cole23a.html
- **vivancocepeda2023geoclip**: GeoCLIP, image–GPS contrastive alignment with hierarchical RFF location encoder. https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b57aaddf85ab01a2445a79c9edc1f4b-Abstract-Conference.html
- **russwurm2024geographic**: spherical-harmonic positional encoding with SIREN networks. *The SH baseline that Slepians generalise; a precision limit near L ≳ 40 in fp32 is noted by Rao et al.* https://openreview.net/forum?id=PudduufFLa (ICLR 2024 spotlight, via api2.openreview.net search) ; arXiv 2310.06743
- **klemmer2025satclip**: SatCLIP, SH+SIREN location encoder trained contrastively against S2 imagery. *Global location embedding; weak on localised tasks, per Rao et al.* https://doi.org/10.1609/aaai.v39i4.32457
- **wu2024torchspatial**: TorchSpatial, location-encoding framework and benchmark. https://proceedings.neurips.cc/paper_files/paper/2024/hash/9449c2d5b0cc8c9a445752f3ff195a1c-Abstract-Datasets_and_Benchmarks_Track.html
- **dhakal2025range**: RANGE, retrieval-augmented multi-resolution geo-embeddings. https://openaccess.thecvf.com/content/CVPR2025/html/Dhakal_RANGE_Retrieval_Augmented_Neural_Fields_for_Multi-Resolution_Geo-Embeddings_CVPR_2025_paper.html
- **cai2025no**: FAIR-Earth dataset plus localised multi-resolution **spherical wavelet** encodings. *An alternative localised basis to Slepians.* https://arxiv.org/abs/2502.06831
- **lane2026sled**: distillation-based location encoder with location as the binding modality. https://arxiv.org/abs/2608.06612
- **rao2026measuring**: intrinsic dimension of Earth representations; motivates choosing the Slepian mode count with the Shannon number. https://arxiv.org/abs/2511.02101
- **rao2026localized**: Slepian (spherical-cap) and hybrid Slepian–SH geographic location encoders. *Direct source of LeLunar's location encoder; the only ML use of Slepian location encoding found.* https://proceedings.mlr.press/v306/rao26a.html ; arXiv 2602.00392 ; https://github.com/arjunarao619/SlepianPosEnc (tag v1.0)

### Slepian theory and tools
- **simons2006spatiospectral**: the spatiospectral concentration problem on the sphere, defining Slepian functions and the Shannon number. https://doi.org/10.1137/S0036144504445765
- **simons2006spherical**: spherical Slepians for polar caps and the polar gap. *Relevant if LeLunar uses polar caps.* https://doi.org/10.1111/j.1365-246X.2006.03065.x
- **wieczorek2005localized**: localised spectral analysis with windowing on the sphere, used in planetary geophysics. https://doi.org/10.1111/j.1365-246X.2005.02687.x
- **bates2017efficient**: efficient Slepian computation for arbitrary regions. https://doi.org/10.1109/TSP.2017.2712122
- **wieczorek2018shtools**: SHTOOLS, a spherical-harmonic and Slepian library. https://doi.org/10.1029/2018GC007529
