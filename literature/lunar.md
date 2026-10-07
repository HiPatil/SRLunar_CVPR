# Lunar / planetary data products and ML on planetary surfaces — verified survey (2026-09-30)

Companion to `lunar.bib` (125 entries). Every entry below was checked against a primary record
(Crossref API, arXiv abs metadata, CVF open-access BibTeX, LPI abstract PDF, publisher/product page).
EO/generic ML works that the lunar FMs build on (TerraMind, MultiMAE, 4M, FlexiViT, FSQ, MAE, SatMAE,
CROMA, AnySat, Galileo, TerraTorch, Prithvi) are deliberately **not** in this file; take them from the EO/SSL bibs.

Author policy in the bib: full lists up to 13 authors; first 12 + `others` beyond that, except the two
NASA-IBM team papers (`patil2026sombench`, `fraccaro2026multimodal`), which list all 22 authors.
Key rule exception: `zuber2013gravityrecovery` (SSR mission paper) vs `zuber2013gravity` (Science field paper) —
both would otherwise be `zuber2013gravity`.

---

## 1. Existing lunar/planetary foundation models (what actually exists as of 2026-09-30)

### Moon — multimodal FMs (all appeared in 2026)

| Model | Who / where | Objective | Data & scale | Downstream | Relation to LeLunar |
|---|---|---|---|---|---|
| **NASA-IBM Lunar Foundation Model ("LFM")** `fraccaro2026multimodal` | Fraccaro, …, **Patil**, …, Roy (IBM Research + NASA IMPACT and partners). arXiv 2609.13283 (2026-09-08); public release 2026-09-10 (NASA/IBM press); NTRS white paper 2026-09-23. Weights Apache-2.0: huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model; code (fine-tuning only, no pretraining code): github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model | **TerraMind masked-token recipe, retrained from scratch**: per-modality FSQ VQ-VAE tokenizers (HF card: nine tokenizers with diffusion decoders), cross-entropy on discrete codes at sampled target positions. ViT-B encoder (768-d, 12 layers) + 12-layer decoder; FlexiViT patch embedding. **Acquisition geometry (incidence, emission, phase, sub-solar anchors, footprint) given as explicit input.** | SomBench: 963,609 WAC tiles (51.2 km, 100 m/px) + 1,000,113 NAC tiles (512 m, 1 m/px). 11 modalities = 9 dense (WAC VIS 5-band, WAC UV 2-band, 60 m DTM/slope/aspect; NAC pan, 3 m DTM/slope/aspect) + optical metadata + a static-map context modality in which **28 static-map variables are averaged over the tile and enter as one token each** (not spatial). 16 H100, 150k steps, ~1.1k GPU-h. | Robbins crater detection @100 m (mAP 0.2581 LoRA vs SwinV2-B 0.2420), NAC crater detection (0.1543 vs SwinV2-B 0.1552, a tie), IMP segmentation (IoU 0.5709 vs ConvNeXt-V2-B 0.5687, a tie), polar ice-prospectivity regression (RMSE 0.0293 vs 0.0377; clear win, but the random-init LFM also beats all but one baseline). | Same team, same data lineage (SomBench layers), same four downstream tasks, and already conditions on illumination geometry. LeLunar cannot claim "first lunar FM", "first multi-resolution NAC+WAC lunar FM", or "first geometry-aware lunar FM". Differences to lean on: latent (JEPA) prediction rather than discrete-token generation; static maps (643 nm mosaic, LOLA roughness, …) as **spatial** tokens rather than tile-averaged scalars. |
| **SomBench** `patil2026sombench` (dataset/benchmark, not a model) | Patil et al., arXiv 2609.13277 (2026-09-08). HF: huggingface.co/datasets/nasa-ibm-ai4science/Sombench-pretraining-data (CC BY 4.0) | — | 30+ co-registered layers, ten instruments, four missions (LRO, Kaguya, GRAIL, Lunar Prospector), 1 m–20 km/px, 90 LTM zones + 2 polar caps, LTM-zone splits, netCDF + Parquet. (IBM press release says "nine instruments".) | Crater (Robbins + NAC hand labels), IMP segmentation (~1,500 NAC images), polar ice prospectivity; ResNet-50 / SwinV2-B baselines. | LeLunar's modality inventory matches SomBench Table 3 row for row (see Section 2), so SomBench is the natural dataset citation (confirm, since the tiling differs). |
| **LunarFM / "Lunar-FM"** `gironamata2026lunarfm` | Girona-Mata, Gawlikowski, Goski, Bardi de Fourtou, Bickel, Moseley, Calzada-Diaz, Kaczmarek, Ramos-Pollán. FDL Europe / Trillium Technologies with Luxembourg Space Agency and ESRIC. arXiv 2607.22408 (2026-07-24); an earlier version was an ICLR 2026 FM4Science workshop paper. lunarfm.trillium.tech | **MultiMAE** (per-modality linear patch projections, shared ViT, per-modality decoders, 85% masking, MSE), 109.8M params, 3 H100, ~500k iterations. | 18 channels from six instruments on three missions: LROC WAC Hapke-normalized mosaic (7 ch), LOLA DEM, Diviner (Treg, rock abundance, Tbol), Mini-RF (CPR, S1), GRAIL (4 ch), Clementine 750 nm. 0.5°×0.5° chips (~15 km) resampled to 112×112; 201,600 chips, 70°S–70°N; diagonal band splits. No NAC, no poles, no slope/aspect/roughness. | Frozen 768-d chip embeddings: qualitative similarity search (L2), random-forest mineral regression (FeO, TiO₂, MgO, Al₂O₃ vs LP-GRS), 10-shot TiO₂ mapping (r = 0.784), USGS 49-unit classification (33.6% top-1, band split). | **Names contrastive learning or JEPA as future work** for disentangling correlated modalities — the gap LeLunar fills. Its similarity search is qualitative only. |
| **Moonstone** `prasad2026moonstone` | Prasad & Mazumder (FMI / DKRZ). ECCV 2026, LNCS pp. 433–450 (arXiv 2607.03644). Data: huggingface.co/datasets/ayushprd/Moonstone; code: github.com/ayushprd/Moonstone | **MG-MAE**: modality-grouped MAE with per-group conv tokenizers, shared ViT-B, missing-modality attention masking, coverage-adaptive masking (60–85%), spectral-continuity regularizer on M³ spectra, plus an InfoNCE term. | 28 channels in 7 groups at 128 ppd (~237 m), global single map (46,080×23,040), random 256×256 crops: WAC morphologic mosaic, LOLA/SLDEM elevation/slope/roughness, Diviner (4), Chandrayaan-1 M³ (8), GRAIL (3), Mini-RF (2), WAC Hapke (4), Clementine 750 nm + LP-GRS TiO₂/FeO. | 6 tasks on 16,200 fixed patches: 49-class geology, 5-class age, FeO/TiO₂ regression, cross-modal thermal regression, mare segmentation, crater (>10 km) segmentation. | Coarse (237 m), no NAC, no illumination geometry. Its "cross-modal thermal regression" is the closest analogue of LeLunar's cross-modal prediction, but done as a downstream probe. |

**Naming collision:** FDL/Trillium brand their model "Lunar-FM" (press) or "LunarFM" (arXiv). The NASA-IBM paper calls its
model "NASA-IBM LFM", and its NTRS PDF is named `Lunar_FM_strives.pdf`. In the paper, write "NASA-IBM LFM [fraccaro2026multimodal]"
and "LunarFM [gironamata2026lunarfm]" so the two are never confused.

### Moon — narrower models and adaptations of generic FMs
- `sander2026moon` (ISPRS J. 2026): one any-to-any transformer over four modalities (grayscale image, DEM, surface normals, albedo) that casts shape-and-albedo-from-shading as multimodal translation; the authors call it a foundation model. Closest prior work to LeLunar's SFS / DTM-reconstruction probes.
- Adapting generic FMs to the Moon: `bauer2026adapting` (Depth Anything V2 fine-tuned on LROC stereo DEMs for height estimation), `grethen2026geometric` (MASt3R distillation for lunar stereo, ECCV 2026), `grethen2025adapting` (StereoLunar rendered stereo set), `bauer2026vision` (OWLv2 + LoRA crater detection for ESA Argonaut), `giannakis2024flexible` (SAM crater detection), `inal2026llavale` (LLaVA fine-tuned on LUCID: 96k high-resolution LROC panchromatic images with captions grounded in GRAIL/LOLA context, plus 81k QA pairs; CVPR 2026 AI4Space workshop).
- Chinese Academy of Sciences Institute of Geochemistry + Alibaba Cloud announced a "lunar science multimodal large language model" in Aug 2024 (crater shape/size/age identification, ">80% accuracy"). **Press release only; no paper, weights or benchmark found** (chinadaily.com.cn, 2024-09-02). Not in the bib.

### Mars
- `purohit2026momo` **MOMO** (CVPR 2026, Kerner lab/JPL): a separate MAE ViT-B per sensor (HiRISE 0.25 m, CTX 5 m, THEMIS 100 m; ~4M samples each, ~12M total), merged by task arithmetic with an "Equal Validation Loss" checkpoint-matching rule. Evaluated on 9 Mars-Bench tasks. Sensors are merged in weight space, not co-registered multimodal inputs.
- `fang2026domain` (JGR: Machine Learning and Computation 2026): CTX-specific ViT FM trained with SSL on millions of CTX images, including similarity search across CTX. The LPSC 2025 precursor abstract (#1374) describes MAE ViT-B on ~7M CTX crops; the journal full text was not accessible (HTTP 403).
- `naik2026marsrecon` MarsRecon (arXiv 2026-09-17): HiRISE MAE (Olympus Mons) plus alignment with text, coordinates and local–global context; retrieval metrics only.
- `purohit2025marsbench` Mars-Bench (NeurIPS 2025 Datasets and Benchmarks): 20 datasets (classification, segmentation, detection), orbital + rover; benchmark, not a model. `panambur2022selfsupervised`: SSL clustering of Curiosity Mastcam textures (rover, CVPRW 2022).

### What we did **not** find (searched 2026-09-30)
- No JEPA / latent-prediction self-supervised model for lunar or planetary orbital data. All three lunar FMs reconstruct pixels or discrete tokens (MultiMAE, MG-MAE, TerraMind-style). The JEPA remote-sensing papers found (e.g., HQ-JEPA, CR-JEPA, ReJEPA) are all Earth observation.
- No lunar FM other than the NASA-IBM LFM that pretrains jointly on NAC 1 m and WAC 100 m; LunarFM and Moonstone are at ≥237 m or on 0.5° chips.
- No lunar FM that uses per-observation (multi-rank) illumination for the same terrain as a learning signal. LFM conditions on geometry per tile, but whether it exploits repeat observations of the same site the way LeLunar's ranks do was not checked.

---

## 2. Which data product each of our modalities most likely comes from, with the citation to use

Basis: SomBench Tables 1 and 3 (same team and pipeline; the LeLunar loader's resolutions and band lists in
`datasets/constants.py` match SomBench Table 3 row by row, with one resolution discrepancy for `MI_SPACE_WEATHERING`),
cross-checked against product READMEs and the product papers.
**Global flag:** LeLunar's `FOOTPRINT_FULLDATA` (10,629 × 60 km footprints, 500 m subtiles) is tiled differently from
the SomBench release (51.2 km / 512 m tiles), so confirm with the data owners that it is SomBench-derived before citing
`patil2026sombench` as the data source.

### Modalities used for LeLunar training
| LeLunar modality (constants) | Most likely product | Cite | Confidence / flag |
|---|---|---|---|
| `WAC_VIS` (415/566/604/643/689 nm, 100 m) and `WAC_UV` (321/360 nm, 500 m), per-rank incidence/emission/phase/sub-solar angles | Individual LROC WAC EDRs, radiometrically calibrated and map-projected by the SomBench pipeline; angles from EDR labels | `robinson2010lunar`, `mahanti2016inflight` (WAC calibration), `speyerer2016preflight` (geometric calibration), `patil2026sombench` | High |
| `DTM_60M`, `SLOPE_60M`, `ASPECT_60M` (60 m) | SLDEM2015 (LOLA + Kaguya TC merge, 60°S–60°N); LOLA-only polar DEM at 60–90°; slope and aspect sin/cos computed by SomBench | `barker2016new` (+ `smith2010lunar`, `haruyama2008global` or `haruyama2012lunar` for the TC input); polar part: SomBench cites `barker2023new`, `barker2025large` | High for SLDEM2015. Exact polar 60 m DEM product not verified. Not GLD100 (`scholten2012gld100` is the WAC stereo DTM and is related only). |
| `WAC_643NM_HR` (100 m) | LROC **WAC empirically normalized** reflectance mosaic (RDR `WAC_EMP`), 643 nm band at 304 ppd; normalized to i=30°, e=0°, g=30° with an empirical photometric function and GLD100 topography | `boyd2012lunar` (photometric function), `wagner2015new` (LROC mosaicked products), `scholten2012gld100` (topography used in normalization), `robinson2010lunar` | Medium-high. Product identity is from the PDS `WAC_EMP_README` (152 ppd 7-band, 304 ppd 643 nm only, 64 ppd 7-band). SomBench itself cites Sato 2017, Sato 2014, Wagner 2015 and Speyerer & Robinson 2013 for this layer. |
| `LOLA_ROUGHNESS` (1 km) | LOLA decameter-scale roughness from the 5-spot pattern (~50 m baseline; std of plane-fit residuals), gridded at 1 km | `neumann2015copernican` (method), `kreslavsky2013lunar` | Medium. SomBench cites exactly these two. `rosenburg2011global` (slopes/roughness from LOLA profiles) is related but probably not the gridded product used. |
| `NAC` (1 m I/F) | LROC NAC EDRs calibrated to I/F | `robinson2010lunar`, `humm2016flight` | High |
| `DTM_3M`, `SLOPE_3M`, `ASPECT_3M` | LROC-team NAC stereo DTMs (native ~2–5 m), resampled to 3 m | `henriksen2017extracting`, `burns2012digital` | High. `beyer2018ames` (ASP) is cited by SomBench for stereo methods, but LROC-team DTMs were not made with ASP. |

### Other dataset modalities
| LeLunar modality | Most likely product | Cite | Confidence / flag |
|---|---|---|---|
| `WAC_NORM_REFL` (7 bands, 500 m) | **Also `WAC_EMP`** (64 ppd, 60°S–60°N; ~137,400 images; i=30°, e=0°, g=30°) — SomBench calls it "empirically normalized" | `boyd2012lunar`, `scholten2012gld100` | **Flag:** the brief calls this "Hapke-normalized". The SomBench description matches the *empirical* WAC_EMP product, not `WAC_HAPKE` (400 m, 70°S–70°N, i=g=60°, e=0°; `sato2014resolved`). LunarFM and Moonstone use the Hapke product. Check the source file before choosing between `boyd2012lunar` and `sato2014resolved`. |
| `WAC_TIO2` (400 m) | LROC WAC TiO₂ from the 321/415 nm ratio, 70°S–70°N | `sato2017lunar` | High |
| `WAC_MORF` (100 m) | WAC Global Morphologic Mosaic (643 nm, high incidence; polar caps from 60°) | `speyerer2011lunar`, `wagner2015new`, `robinson2012exploring` | High |
| `MI_NORM_REFL` (8 bands, 60 m) | Kaguya MI topographically corrected reflectance mosaic (512 ppd) | `ohtake2008performance`, `lemelin2016global`, `lemelin2015lunar` | Medium-high |
| `MI_MINERALOGY` (CPX, OPX, olivine, plagioclase, plag. grain size, FeO, OMAT; 60 m) | Kaguya MI 512 ppd mineral/FeO/OMAT maps | `lemelin2016global`, `lemelin2015lunar`, `lemelin2019compositions` | High |
| `MI_SPACE_WEATHERING` (smFe, mpFe, npFe) | Trang & Lucey radiative-transfer space-weathering maps | `trang2019improved` | High for the product. **Resolution mismatch:** SomBench says 1 km/px; `constants.py` declares 60 m. |
| `KAGUYA_SP_*` (olivine, plagioclase, HCP, LCP, npFe, FeO, OMAT; 1 km, polar) | Kaguya SP polar compositional maps, poleward of 50°, 1 km/px (Zenodo 10.5281/zenodo.5847000) | **`lemelin2022compositional`** (+ `haruyama2008global`, `yamamoto2014calibration` for the instrument) | High: the layer set and resolution match exactly. SomBench cites only the instrument papers, so this citation corrects an omission. |
| `DIVINER_ROCK`, `DIVINER_TREG` (240 m, 70°S–70°N) | Diviner rock abundance and nighttime regolith-temperature anomaly with topographic removal | **`powell2023high`** (method heritage `bandfield2011lunar`); instrument `paige2010lunar` | High. The 240 m maps are Powell 2023, not Bandfield 2011's original maps. |
| `DIVINER_HPAR` (240 m) | Diviner H-parameter (thermal inertia proxy) | `hayne2017global` | High |
| `DIVINER_TBOL` (conditioning; 24 sub-solar-longitude bins; 0.5° ≈ 15 km) | Diviner global gridded bolometric temperatures | `williams2017global`, `paige2010lunar` | High |
| `DIVINER_TBOL_SUMMER/WINTER` (240 m, polar) | Diviner seasonal polar temperatures | `williams2019seasonal` | High |
| `DIVINER_ICE` (240 m) | Ice stability depth | **`schorghofer2020mapping`** (context: `paige2010diviner`) | High |
| `MINI_RF_RADAR` (S1 reflectivity, CPR; 90 m) | Mini-RF monostatic S-band mosaics, orthorectified and topographically normalized | `fassett2024improved`, `nozette2010lunar`, `raney2011lunar` (+ `cahill2014miniature`, `raney2007hybrid`) | Medium-high (follows SomBench; the 90 m mosaic spec was not independently checked) |
| `LOLA_AVGVISIB` (120 m, 75–90°) | LOLA average solar visibility over a precession cycle | `mazarico2011illumination` (+ `speyerer2013persistently`) | High |
| `LOLA_PSR` (20 m, 80–90°) | LOLA-derived PSR map | SomBench cites `barker2023new`, `barker2025large`, `mazarico2011illumination` | Medium (exact 20 m PSR product not verified) |
| `LOLA_ALBEDO` (1 km, 50–90°) | LOLA 1064 nm normal albedo | `lemelin2016improved` | High |
| `GRAIL_GRAVITY` (conditioning; 20 km) | GRAIL free-air gravity disturbance | `zuber2013gravity`, `zuber2013gravityrecovery`; spherical-harmonic model is one of `lemoine2014grgm900c` / `konopliv2014high` / `goossens2020high` | **Low-medium: which GRAIL model or degree underlies the map was not established.** |
| `LP_HYDROGEN` (conditioning; 15 km, polar) | Lunar Prospector Neutron Spectrometer hydrogen abundance | `lawrence2022global` (+ `feldman1998fluxes`, `feldman2001evidence`, `feldman1999lunar`, `binder1998lunar`) | Medium-high (follows SomBench) |
| `GEOMAPS` (60 m; never sampled by LeLunar) | USGS Unified Geologic Map of the Moon, 1:5M | `fortezzo2020release` | High |
| Split scheme | Lunar Transverse Mercator zones / polar stereographic caps | `mcclernan2025lunar` | High |
| Crater labels | Robbins global database | `robbins2019new` (catalog comparison: `heyer2023comparative`) | High |
| IMP labels / context | IMP catalogs | `braden2014evidence` (70 IMPs), `qiao2020lunar` (updated catalog), `hargitai2025clusters` (GIS catalog; cited by SomBench/LFM) | The provenance of the IMP label masks is per SomBench (~1,500 NAC IMP images) and was not re-checked |
| Ice-prospectivity labels | VIPER-site prospectivity model | `coyan2025prospectivity` (+ `schorghofer2020mapping`, `mazarico2011illumination`, `lawrence2022global`) | Per SomBench/LFM |

---

## 3. Unverified items, corrections, and caveats

- **Brief vs record:** "Chin et al. 2010" does not exist as an overview. The LRO overview is Chin et al. **2007** (SSR 129) or Vondrak et al. 2010 (SSR 150). The "Lemelin 2016" global MI maps are an **LPSC abstract** (#2994), not a journal paper. GRAIL models are Konopliv/Lemoine **2014** (GRL). The requested Bandfield 2011 and Paige 2010 (Science) are verified, but the dataset's layers come from Powell 2023 and Schorghofer & Williams 2020.
- **"WAC Hapke-normalized reflectance"** is probably a misnomer for the empirically normalized `WAC_EMP` product (Section 2). Unresolved until the source file is checked.
- **Uncertain product identity:** GRAIL model behind the 20 km disturbance map; exact LOLA polar 60 m DEM and 20 m PSR products; LOLA roughness product (Neumann 2015 vs Kreslavsky 2013 gridding); `MI_SPACE_WEATHERING` resolution (1 km vs 60 m); the Mini-RF 90 m mosaic spec.
- **Whether LeLunar's footprint dataset is SomBench-derived** (the layers match; the tiling and counts do not).
- **Problems in sibling-paper bibliographies (useful for the team):** the NASA-IBM LFM paper's ref [2], "Robinson et al., Lunar Reconnaissance Orbiter Camera: First results, Science 2010", could not be found in Crossref — do not propagate it. That paper also dates Henriksen et al. to 2016 (print is Icarus 283, 2017) and Paige et al. to 2009 (print is SSR 150, 2010). SomBench ref [46] (Raney et al., Proc. IEEE 99) is 2011, not 2010.
- **Venue details not fully resolvable:** LNCS volume numbers for the ECCV 2026 chapters (`prasad2026moonstone`, `grethen2026geometric`) are not in Crossref, so they are omitted; Moonstone is in "Proceedings, Part XVIII". `bauer2026vision` (IEEE TAES) is early access with no volume yet. `fang2026domain`: the publisher page returned 403, so metadata comes from Crossref and the method details from the LPSC 2025 precursor abstract. `wagstaff2018deep` page range (7867–7872) comes from Semantic Scholar (the AAAI OJS page shows none).
- **Name spellings taken from the records:** `haruyama2012lunar` — the LPSC PDF prints "Koii Matsumoto" (almost certainly Koji), so the bib uses "K. Matsumoto". `grethen2026geometric` — arXiv metadata gives "Florient Chouteau"; Crossref lists only the first author. `gironamata2026lunarfm` — "Gautier Bardi de Fourtou" is written in First-von-Last form so it prints as in the paper.
- **Press numbers vs paper numbers:** NASA/secondary coverage says "up to 23%" improvement; the IBM newsroom says up to 22% lower ice RMSE and "nearly 19%" better crater detection at a given resolution. Quote the arXiv tables, not the press releases.
- **Not included** (could not be verified to our standard, or out of scope): Park et al. 2025 Nature (GRAIL; cited by SomBench), Mazarico et al. 2012 (LRO orbits; cited in the WAC_EMP README), the Chinese IGCAS/Alibaba lunar LLM (press only), and the LunarFM ICLR 2026 workshop version (seen only as an iclr.cc listing, "LunarFM: a multimodal representation of the Moon's surface"; cite the arXiv version).

---

## 4. Per-entry notes (key — what it is | how we cite it | how verified)

### Missions and instruments
- `chin2007lunar` — LRO mission and instrument-suite overview (SSR 129, 2007) | Mission citation for all LRO data | Crossref https://doi.org/10.1007/s11214-007-9153-y
- `vondrak2010lunar` — LRO observations for exploration and science (SSR 150, 2010) | Alternative/companion LRO mission citation | Crossref https://doi.org/10.1007/s11214-010-9631-5
- `robinson2010lunar` — LROC (NAC + WAC) instrument overview | Source instrument for WAC VIS/UV, NAC and WAC mosaics | Crossref https://doi.org/10.1007/s11214-010-9634-2
- `smith2010lunar` — LOLA instrument and investigation | Instrument citation for LOLA topography, roughness, albedo, illumination | Crossref https://doi.org/10.1007/s11214-009-9512-y
- `paige2010lunar` — Diviner instrument paper (SSR 150) | Instrument citation for all Diviner layers | Crossref https://doi.org/10.1007/s11214-009-9529-2
- `nozette2010lunar` — Mini-RF technology demonstration on LRO | Instrument citation for Mini-RF S1/CPR | Crossref https://doi.org/10.1007/s11214-009-9607-5
- `raney2011lunar` — Mini-RF hybrid-polarimetric architecture and first results (Proc. IEEE 2011) | Radar-mode/CPR definition | Crossref https://doi.org/10.1109/JPROC.2010.2084970
- `raney2007hybrid` — hybrid-polarity SAR architecture | Background for CPR/S1 from hybrid polarimetry | Crossref https://doi.org/10.1109/TGRS.2007.895883
- `cahill2014miniature` — Mini-RF global observations of the Moon | Global Mini-RF coverage and mosaic context | Crossref https://doi.org/10.1016/j.icarus.2014.07.018
- `kato2010kaguya` — Kaguya/SELENE mission overview | Mission citation for TC/MI/SP layers | Crossref https://doi.org/10.1007/s11214-010-9678-3
- `ohtake2008performance` — Kaguya Multiband Imager performance and objectives | Instrument citation for MI reflectance/mineralogy | Crossref https://doi.org/10.1186/BF03352789
- `haruyama2008global` — LISM (TC, MI, SP) mapping experiment on SELENE | Instrument citation for Kaguya TC (input to SLDEM2015) and SP | Crossref https://doi.org/10.1186/BF03352788
- `yamamoto2014calibration` — Kaguya Spectral Profiler NIR2 calibration | SP instrument/calibration citation for polar SP maps | Crossref https://doi.org/10.1109/TGRS.2014.2304581
- `zuber2013gravityrecovery` — GRAIL mission paper (SSR 178) | Mission citation for gravity conditioning | Crossref https://doi.org/10.1007/s11214-012-9952-7
- `zuber2013gravity` — gravity field of the Moon from GRAIL (Science 339) | Primary GRAIL result citation | Crossref https://doi.org/10.1126/science.1231507
- `binder1998lunar` — Lunar Prospector overview (Science 281) | Mission citation for LP hydrogen | Crossref https://doi.org/10.1126/science.281.5382.1475
- `feldman1999lunar` — LP gamma-ray and neutron spectrometers (NIM A) | Instrument citation for LP neutron data | Crossref https://doi.org/10.1016/S0168-9002(98)00934-6
- `mahanti2016inflight` — in-flight radiometric calibration of LROC WAC | Calibration basis for our WAC VIS/UV I/F | Crossref https://doi.org/10.1007/s11214-015-0197-0
- `humm2016flight` — flight calibration of LROC NAC | Calibration basis for NAC I/F | Crossref https://doi.org/10.1007/s11214-015-0201-8
- `speyerer2016preflight` — LROC geometric calibration | Map-projection/camera-model basis for WAC/NAC tiles | Crossref https://doi.org/10.1007/s11214-014-0073-3

### Derived map products
- `scholten2012gld100` — GLD100 WAC stereo 100 m DTM | Related topography; topography source for WAC_EMP normalization | Crossref https://doi.org/10.1029/2011JE003926 (article no. E00H17 confirmed via ASU record/search)
- `barker2016new` — SLDEM2015 (LOLA + Kaguya TC, 60°S–60°N) | Source of `DTM_60M`/slope/aspect | Crossref https://doi.org/10.1016/j.icarus.2015.07.039
- `haruyama2012lunar` — Kaguya TC global DTM dataset (LPSC 2012 #1200) | TC stereo input to SLDEM2015 | LPI PDF https://www.lpi.usra.edu/meetings/lpsc2012/pdf/1200.pdf
- `barker2023new` — new LOLA view of the lunar south pole | Polar LOLA topography/PSR context | Crossref https://doi.org/10.3847/PSJ/acf3e1
- `rosenburg2011global` — global LOLA slopes and roughness | Related roughness background (not the gridded layer) | Crossref https://doi.org/10.1029/2010JE003716
- `kreslavsky2013lunar` — LOLA roughness maps, scale dependence | Roughness product/method citation | Crossref https://doi.org/10.1016/j.icarus.2013.04.027
- `neumann2015copernican` — LOLA decameter-scale (5-spot) roughness (LPSC 2015 #2218) | Method behind the 1 km `LOLA_ROUGHNESS` layer | LPI PDF https://www.hou.usra.edu/meetings/lpsc2015/pdf/2218.pdf
- `barker2025large` — large-scale polar roughness from LOLA | Polar topography/PSR context (cited by SomBench) | Crossref https://doi.org/10.3847/PSJ/adbc9d
- `mazarico2011illumination` — polar illumination from LOLA topography | Source of `LOLA_AVGVISIB`; PSR/shadow context for the shadow probe | Crossref https://doi.org/10.1016/j.icarus.2010.10.030
- `speyerer2013persistently` — persistently illuminated polar regions from WAC | Polar illumination context; polar WAC mosaics | Crossref https://doi.org/10.1016/j.icarus.2012.10.010
- `lemelin2016improved` — calibrated LOLA 1064 nm reflectance | Source of `LOLA_ALBEDO` | Crossref https://doi.org/10.1016/j.icarus.2016.02.006
- `speyerer2011lunar` — WAC global morphologic map (LPSC 2011 #2387) | Source of `WAC_MORF` | LPI PDF https://www.lpi.usra.edu/meetings/lpsc2011/pdf/2387.pdf
- `robinson2012exploring` — overview of LROC products (ISPRS Archives) | Companion citation for WAC mosaics | Crossref https://doi.org/10.5194/isprsarchives-XXXIX-B4-501-2012
- `wagner2015new` — new LROC mosaicked products incl. polar (LPSC 2015 #1473) | Mosaic citation for `WAC_643NM_HR` / polar `WAC_MORF` | LPI PDF https://www.hou.usra.edu/meetings/lpsc2015/pdf/1473.pdf
- `boyd2012lunar` — empirical WAC photometric function (LPSC 2012 #2795) | Normalization behind `WAC_643NM_HR` and `WAC_NORM_REFL` (WAC_EMP) | LPI PDF https://www.lpi.usra.edu/meetings/lpsc2012/pdf/2795.pdf; product README https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001/DATA/MDR/WAC_EMP/WAC_EMP_README.TXT
- `sato2014resolved` — resolved Hapke parameter maps from WAC | Cite only if the Hapke-normalized mosaic (WAC_HAPKE) was used; photometric-modeling background | Crossref https://doi.org/10.1002/2013JE004580
- `sato2017lunar` — WAC TiO₂ from UV/Vis ratio | Source of `WAC_TIO2` | Crossref https://doi.org/10.1016/j.icarus.2017.06.013
- `lemelin2015lunar` — MI-based mineralogy method (central peaks) | Method for `MI_MINERALOGY` | Crossref https://doi.org/10.1002/2014JE004778
- `lemelin2016global` — global MI 512 ppd minerals/FeO/OMAT maps (LPSC 2016 #2994) | Product citation for `MI_MINERALOGY` (and MI reflectance) | LPI PDF https://www.hou.usra.edu/meetings/lpsc2016/pdf/2994.pdf
- `lemelin2019compositions` — MI spectral analysis of basin inner rings | Additional MI mineralogy citation (SomBench cites it) | Crossref https://doi.org/10.1016/j.pss.2018.10.003
- `trang2019improved` — MI space-weathering (smFe/mpFe/npFe) maps | Source of `MI_SPACE_WEATHERING` | Crossref https://doi.org/10.1016/j.icarus.2018.11.014
- `lemelin2022compositional` — Kaguya SP + LOLA polar compositional maps (1 km, >50°) | Source of `KAGUYA_SP_*` | Crossref https://doi.org/10.3847/PSJ/ac532c; abstract via iopscience
- `fassett2024improved` — improved orthorectification/topographic reduction of Mini-RF S-band | Source of `MINI_RF_RADAR` mosaics | Crossref https://doi.org/10.3847/PSJ/ad0a61
- `williams2017global` — Diviner global surface temperatures | Source of `DIVINER_TBOL` (conditioning) | Crossref https://doi.org/10.1016/j.icarus.2016.08.012
- `williams2019seasonal` — Diviner seasonal polar temperatures | Source of `DIVINER_TBOL_SUMMER/WINTER` | Crossref https://doi.org/10.1029/2019JE006028
- `schorghofer2020mapping` — ice storage/stability mapping with time-dependent temperatures | Source of `DIVINER_ICE`; ice-prospectivity context | Crossref https://doi.org/10.3847/PSJ/abb6ff
- `powell2023high` — high-res Diviner nighttime temperature and rock abundance | Source of `DIVINER_ROCK`, `DIVINER_TREG` | Crossref https://doi.org/10.1029/2022JE007532
- `hayne2017global` — Diviner regolith thermophysical properties (H-parameter) | Source of `DIVINER_HPAR` | Crossref https://doi.org/10.1002/2017JE005387
- `bandfield2011lunar` — original Diviner rock abundance / regolith temperature method | Method heritage for rock/Treg layers (requested) | Crossref https://doi.org/10.1029/2011JE003866
- `paige2010diviner` — Diviner cold traps at the south pole (Science 330) | Ice/cold-trap context for the ice probe (requested) | Crossref https://doi.org/10.1126/science.1187726
- `konopliv2014high` — JPL GRAIL gravity fields (primary + extended mission) | Candidate model behind `GRAIL_GRAVITY` | Crossref https://doi.org/10.1002/2013GL059066
- `lemoine2014grgm900c` — GSFC GRGM900C degree-900 gravity model | Candidate model behind `GRAIL_GRAVITY` | Crossref https://doi.org/10.1002/2014GL060027
- `goossens2020high` — high-res GRAIL models and crustal density | Candidate model (SomBench cites it) | Crossref https://doi.org/10.1029/2019JE006086
- `feldman1998fluxes` — LP epithermal-neutron evidence for polar ice (Science 281) | Discovery citation for LP hydrogen | Crossref https://doi.org/10.1126/science.281.5382.1496
- `feldman2001evidence` — LP evidence for water ice near the poles (JGR 2001) | Classic polar-hydrogen map citation | Crossref https://doi.org/10.1029/2000JE001444
- `lawrence2022global` — global hydrogen abundances from LP | Source of `LP_HYDROGEN` | Crossref https://doi.org/10.1029/2022JE007197
- `fortezzo2020release` — USGS Unified Geologic Map 1:5M (LPSC 2020 #2760) | Source of `GEOMAPS`; LunarFM/Moonstone classification labels | LPI PDF https://www.hou.usra.edu/meetings/lpsc2020/pdf/2760.pdf
- `henriksen2017extracting` — LROC NAC stereo DTM production and accuracy | Source of `DTM_3M`/slope/aspect | Crossref https://doi.org/10.1016/j.icarus.2016.05.012
- `burns2012digital` — NAC DEMs and derived products (ISPRS Archives) | Companion NAC DTM citation | Crossref https://doi.org/10.5194/isprsarchives-XXXIX-B4-483-2012
- `beyer2018ames` — Ames Stereo Pipeline | Tooling citation if ASP is used for SFS/DTM work | Crossref https://doi.org/10.1029/2018EA000409
- `mcclernan2025lunar` — USGS lunar grids/projections incl. LTM (TM 11-E1) | Citation for the LTM-strip splits | Crossref https://doi.org/10.3133/tm11E1
- `wagner2024where` — best practices for accurate NAC coordinates | Co-registration/geolocation caveat for NAC–DTM pairing | Crossref https://doi.org/10.3847/PSJ/ad54c6

### Label sources and science context
- `robbins2019new` — global lunar crater database (>1–2 km, ~2M craters) | Crater-detection labels | Crossref https://doi.org/10.1029/2018JE005592
- `heyer2023comparative` — comparison of global lunar crater catalogs (OpenCraterTool) | Label-quality caveat for Robbins | Crossref https://doi.org/10.1016/j.pss.2023.105687
- `braden2014evidence` — the original IMP catalog (70 IMPs; model ages <100 Ma for Ina, Cauchy-5, Sosigenes) | Motivation/definition for IMP segmentation | Crossref https://doi.org/10.1038/ngeo2252 (IMP count confirmed via secondary sources)
- `qiao2020lunar` — IMP classification and updated catalog | IMP task context | Crossref https://doi.org/10.1029/2019JE006362
- `hargitai2025clusters` — GIS catalog of irregular-patch clusters | IMP catalog cited by SomBench/LFM | Crossref https://doi.org/10.1016/j.icarus.2024.116439
- `coyan2025prospectivity` — VIPER-site ice-prospectivity model | Ice-prospectivity label source | Crossref https://doi.org/10.3847/PSJ/adbc6c
- `li2018direct` — M³ evidence for surface-exposed polar water ice (PNAS) | Ice context | Crossref https://doi.org/10.1073/pnas.1802345115
- `fisher2017evidence` — LOLA reflectance + Diviner temperature evidence for surface ice | Justifies LOLA albedo + Diviner as ice cues | Crossref https://doi.org/10.1016/j.icarus.2017.03.023

### Lunar / planetary foundation and SSL models
- `fraccaro2026multimodal` — NASA-IBM Lunar FM (Section 1) | Main prior lunar FM; must be compared against | arXiv https://arxiv.org/abs/2609.13283 (+ HTML full text, HF model card, NASA and IBM release pages, NTRS 20260008697)
- `patil2026sombench` — SomBench dataset/benchmark (Section 1) | Dataset and benchmark citation | arXiv https://arxiv.org/abs/2609.13277 (+ HTML full text incl. Table 3 and bibliography; HF dataset card)
- `gironamata2026lunarfm` — LunarFM (FDL/Trillium MultiMAE) | Prior lunar FM; similarity-search prior art; JEPA suggested as future work | arXiv https://arxiv.org/abs/2607.22408 (+ HTML full text; lunarlab.ai; Luxembourg Space Agency release)
- `prasad2026moonstone` — Moonstone MG-MAE and benchmark (ECCV 2026) | Prior lunar FM and benchmark | arXiv https://arxiv.org/abs/2607.03644; Crossref https://doi.org/10.1007/978-3-032-37517-9_24 (pp. 433–450)
- `sander2026moon` — unified multimodal lunar transformer (image/DEM/normals/albedo) | Prior multimodal lunar reconstruction/SAfS work | Crossref https://doi.org/10.1016/j.isprsjprs.2026.04.008; arXiv https://arxiv.org/abs/2505.05644
- `purohit2026momo` — MOMO Mars multi-sensor FM (CVPR 2026) | Planetary FM precedent (MAE + model merging) | CVF https://openaccess.thecvf.com/content/CVPR2026/html/Purohit_MOMO_Mars_Orbital_MOdel_Foundation_Model_for_Mars_Orbital_Applications_CVPR_2026_paper.html; arXiv https://arxiv.org/abs/2604.02719
- `fang2026domain` — CTX Mars vision FM with SSL (JGR:MLC) | Planetary SSL precedent; similarity search at planetary scale | Crossref https://doi.org/10.1029/2025JH000827; precursor https://www.hou.usra.edu/meetings/lpsc2025/pdf/1374.pdf
- `naik2026marsrecon` — HiRISE MAE + text/geo alignment | Recent Mars SSL/retrieval | arXiv https://arxiv.org/abs/2609.22379
- `purohit2025marsbench` — Mars-Bench (20 datasets) | Planetary FM evaluation-benchmark precedent | Crossref https://doi.org/10.52202/085713-4098; arXiv https://arxiv.org/abs/2510.24010
- `panambur2022selfsupervised` — SSL clustering of Mastcam terrain textures | Early planetary SSL (rover) | CVF https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Panambur_Self-Supervised_Learning_To_Guide_Scientifically_Relevant_Categorization_of_Martian_Terrain_CVPRW_2022_paper.html

### Lunar ML: craters, rocks, shadows, anomalies, ice, topography
- `silburt2019lunar` — DeepMoon: U-Net crater detection on DEMs | Prior crater-detection ML | Crossref https://doi.org/10.1016/j.icarus.2018.06.022
- `lee2019automated` — CNN crater detection on Mars | Prior crater ML (Mars) | Crossref https://doi.org/10.1016/j.pss.2019.03.008
- `delatte2019automated` — review of CNN-era crater detection | Crater-detection review | Crossref https://doi.org/10.1016/j.asr.2019.07.017
- `wang2020effective` — CNN lunar crater recognition | Prior crater ML (cited by LFM) | Crossref https://doi.org/10.3390/rs12172694
- `yang2020lunar` — Chang'E crater identification + age estimation by transfer learning | Prior large-scale lunar crater ML | Crossref https://doi.org/10.1038/s41467-020-20215-y
- `tewari2022automated` — crater detection from co-registered optical + DEM + slope | Closest prior multi-input crater detection (our inputs are similar) | Crossref https://doi.org/10.1016/j.pss.2022.105500
- `tewari2023deep` — review of DL crater-detection systems (arXiv) | Crater-detection review | arXiv https://arxiv.org/abs/2310.07727
- `lagrassa2023yololens` — super-resolution-assisted YOLO crater detection | Prior crater ML (cited by LFM) | Crossref https://doi.org/10.3390/rs15051171
- `giannakis2024flexible` — SAM-based universal crater detection | Generic-FM baseline for craters | Crossref https://doi.org/10.1016/j.icarus.2023.115797
- `chen2024impact` — review of crater recognition methods | Crater-detection review | Crossref https://doi.org/10.1007/s11430-023-1284-9
- `sinha2025automated` — review of lunar DL crater detection (Chandrayaan-2 TMC-2) | Recent review (optional) | Crossref https://doi.org/10.1016/j.rines.2025.100094
- `ma2026deep` — CNN/ResNet/YOLO crater framework for Moon and Mars | Recent crater ML (optional) | Crossref https://doi.org/10.1038/s44453-026-00036-x
- `bauer2026vision` — OWLv2 VLM + LoRA crater detection (ESA Argonaut) | Hazard-oriented crater ML adapting a generic FM | Crossref https://doi.org/10.1109/TAES.2026.3725640; arXiv https://arxiv.org/abs/2601.07795
- `fang2026craterbench` — CraterBench-R instance-level crater retrieval (CVPRW 2026) | Retrieval prior art for our similarity search; in-domain SSL ViTs win | CVF https://openaccess.thecvf.com/content/CVPR2026W/EarthVision/html/Fang_CraterBench-R_Instance-Level_Crater_Retrieval_for_Planetary_Scale_CVPRW_2026_paper.html
- `bickel2019automated` — CNN detection of lunar rockfalls in NAC | Prior NAC-scale ML | Crossref https://doi.org/10.1109/TGRS.2018.2885280
- `bickel2020impacts` — global rockfall mapping (impacts drive rockfalls) | Science use of NAC ML at global scale | Crossref https://doi.org/10.1038/s41467-020-16653-3
- `bickel2021peering` — HORUS PSR image enhancement | Shadow/PSR ML prior art for the shadow probe | Crossref https://doi.org/10.1038/s41467-021-25882-z
- `aussel2025global` — global lunar boulder map from NAC with DL | Recent NAC-scale ML at global scale | Crossref https://doi.org/10.1029/2025JE008981
- `moseley2020unsupervised` — VAE trained on 9+ yr of Diviner temperature curves that disentangles five thermophysical factors | Prior unsupervised representation learning on lunar data | Crossref https://doi.org/10.3847/PSJ/ab9a52 (abstract read via Crossref)
- `lesnikowski2024automated` — VAE anomaly detector on NAC patches; best on fresh craters, IMPs and pits (AP gains of 2–327× over random) | Prior ML on IMPs; retrieval prior art | Crossref https://doi.org/10.1109/JSTARS.2024.3369101; arXiv https://arxiv.org/abs/2403.07424 (full text checked for the IMP class)
- `wang2025identification` — deep-learning, pixel-level identification of H₂O ice in the south polar region (Icarus 2025) | Prior ML for polar ice | Crossref https://doi.org/10.1016/j.icarus.2025.116682 (metadata only; publisher page returned 403, so the method was not read)
- `alexandrov2018multiview` — multiview SFS in ASP | SFS downstream reference | Crossref https://doi.org/10.1029/2018EA000390
- `grumpe2014construction` — lunar DEMs from reflectance modelling (SfS) | Classical lunar SfS reference | Crossref https://doi.org/10.1016/j.asr.2013.09.036
- `wu2018construction` — SAfS pixel-level DEMs constrained by low-res DEM | Classical SAfS reference | Crossref https://doi.org/10.1016/j.isprsjprs.2017.03.007
- `chen2022cnn` — CNN single-view NAC topography constrained by SLDEM | Learned SFS/DTM prior art | Crossref https://doi.org/10.1109/JSTARS.2022.3214926
- `chen2024elunardtmnet` — ELunarDTMNet single-view lunar DTM | Learned DTM reconstruction prior art | Crossref https://doi.org/10.1109/TGRS.2024.3501153
- `liu2022generative` — GAN pixel-scale lunar DEM from image + low-res DEM | Learned DTM prior art | Crossref https://doi.org/10.3390/rs14215420
- `bauer2026adapting` — Depth Anything V2 fine-tuned for lunar height | Generic-FM baseline for DTM probe | arXiv https://arxiv.org/abs/2609.02448
- `grethen2025adapting` — StereoLunar rendered stereo dataset + MASt3R fine-tuning | Lunar 3D benchmark (synthetic) | CVF https://openaccess.thecvf.com/content/ICCV2025W/3D-VAST/html/Grethen_Adapting_Stereo_Vision_From_Objects_To_3D_Lunar_Surface_Reconstruction_ICCVW_2025_paper.html
- `grethen2026geometric` — MASt3R distillation for lunar 3D (ECCV 2026) | Lunar 3D FM adaptation | arXiv https://arxiv.org/abs/2607.01851; Crossref https://doi.org/10.1007/978-3-032-37432-5_30
- `grethen2026moonanything` — MoonAnything rendered geometry/photometry benchmark | Lunar photometric/3D benchmark (synthetic) | Crossref https://doi.org/10.1145/3793853.3799814
- `inal2026llavale` — LLaVA-LE lunar VLM + LUCID dataset | Lunar VLM prior work | CVF https://openaccess.thecvf.com/content/CVPR2026W/AI4Space/html/Inal_LLaVA-LE_Large_Language-and-Vision_Assistant_for_Lunar_Exploration_CVPRW_2026_paper.html
- `moghe2020deep` — DL hazard detection for lunar landing | Landing/hazard ML | Crossref https://doi.org/10.1007/s40295-020-00239-8
- `downes2020lunar` — CNN crater detection for lunar terrain-relative navigation | Landing/navigation ML | Crossref https://doi.org/10.23919/ACC45564.2020.9147595
- `silvestrini2022optical` — CNN crater detector for lunar-landing optical navigation | Landing/navigation ML | Crossref https://doi.org/10.1016/j.ast.2022.107503
- `wang2026marsretrieval` — MarsRetrieval VLM retrieval benchmark | Retrieval-benchmark prior art | arXiv https://arxiv.org/abs/2602.13961

### Mars ML datasets and systems
- `swan2021ai4mars` — AI4Mars rover terrain segmentation dataset | Mars ML dataset precedent | CVF https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/html/Swan_AI4MARS_A_Dataset_for_Terrain-Aware_Autonomous_Driving_on_Mars_CVPRW_2021_paper.html
- `wagstaff2018deep` — Deep Mars: CNN classification for the PDS Imaging Atlas | Deployed planetary content-based search precedent | Crossref https://doi.org/10.1609/aaai.v32i1.11404 (pages via Semantic Scholar API)
- `wagstaff2021mars` — three years of deployed Mars image classification | Deployed planetary ML | Crossref https://doi.org/10.1609/aaai.v35i17.17784
- `vasu2024interactive` — interactive content-based Mars image search | Similarity-search precedent | Crossref https://doi.org/10.1609/aaai.v38i21.30338
- `wilhelm2020domars16k` — DoMars16k CTX landform dataset | Mars benchmark dataset | Crossref https://doi.org/10.3390/rs12233981
- `liu2024marsscapes` — MarsScapes panoramic dataset + UDAFormer | Mars terrain-segmentation dataset (requested) | Crossref https://doi.org/10.1109/TGRS.2023.3343109
- `purohit2024conequest` — ConeQuest cone-segmentation benchmark | Mars benchmark (part of Mars-Bench) | Crossref https://doi.org/10.1109/WACV57701.2024.00592

### Reviews / perspectives
- `azari2021integrating` — perspectives on ML for planetary science (BAAS 53(4)) | General motivation for planetary ML | Crossref https://doi.org/10.3847/25c2cfeb.aa328727
- `helbert2022machine` — edited book *Machine Learning for Planetary Science* (Elsevier 2022) | General reference | Elsevier/ScienceDirect book page https://www.sciencedirect.com/book/9780128187210/machine-learning-for-planetary-science (editors, ISBN)
