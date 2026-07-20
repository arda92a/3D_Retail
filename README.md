# 3D Retail Analytics Research

A curated collection of papers and repositories for semantic 3D reconstruction, novel view synthesis, and synthetic data generation applied to retail shelf analytics.

---

## Repositories

### 3D Gaussian Splatting

| Repository | Description |
|------------|-------------|
| [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | Original 3DGS implementation (Inria / SIGGRAPH 2023) |
| [lkeab/gaussian-grouping](https://github.com/lkeab/gaussian-grouping) | Segment and edit anything in 3D — semantic identity encoding |
| [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | Efficient 3DGS CUDA kernels |
| [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio) | Modular NeRF / 3DGS training and rendering framework |

### Camera Pose Estimation & SfM

| Repository | Description |
|------------|-------------|
| [colmap/gluemap](https://github.com/colmap/gluemap) | Global SfM + feedforward reconstruction — robust to repeating textures |
| [colmap/colmap](https://github.com/colmap/colmap) | Classic Structure-from-Motion and MVS pipeline |
| [naver/dust3r](https://github.com/naver/dust3r) | DUSt3R — geometric 3D vision without calibration |
| [naver/mast3r](https://github.com/naver/mast3r) | MASt3R — dense matching for DUSt3R |
| [mv-dust3rp](https://mv-dust3rp.github.io/) | MV-DUSt3R+ — sparse-view reconstruction in 2 seconds (CVPR 2025 Oral) |

### Reading Lists

| Resource | Link |
|----------|------|
| MrNeRF's Awesome 3DGS Paper List | [mrnerf.github.io](https://mrnerf.github.io/awesome-3D-gaussian-splatting/) |
| NAVER LABS 3D Foundation Models | [europe.naverlabs.com](https://europe.naverlabs.com/research/3d-foundation-models/) |

---

## Papers

### Core 3DGS & NeRF

| Paper | Venue | Link |
|-------|-------|------|
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering | SIGGRAPH 2023 | [arXiv](https://arxiv.org/abs/2308.04079) |
| NeRF: Representing Scenes as Neural Radiance Fields | ECCV 2020 | [arXiv](https://arxiv.org/abs/2003.08934) |
| NeRF: A Comprehensive Review | — | [arXiv](https://arxiv.org/html/2210.00379v6) |
| 3DGS vs NeRF: Side-by-Side Reconstruction Comparison | Research Square | [PDF](https://assets-eu.researchsquare.com/files/rs-7300179/v1_covered_60fc3280-68ec-47c4-ae1d-228296ea12c4.pdf) |
| Evaluating 3DGS and NeRF Under Various Filming Techniques | — | [PDF](https://research.utwente.nl/files/542456748/preprints202504.1068.v1.pdf) |
| From Fields to Splats: Cross-Domain Survey of Neural Scene Representations | — | [arXiv](https://arxiv.org/html/2509.23555v1) |
| VCR-GauS: View Consistent Depth-Normal Regularizer for Gaussian Surface Reconstruction | NeurIPS 2024 | [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/fc9f83d9925e6885e8f1ae1e17b3c44b-Paper-Conference.pdf) |
| From Chaos to Clarity: 3DGS in the Dark | OpenReview | [Link](https://openreview.net/forum?id=lWHe7pmk7C) |

### Semantic Scene Understanding in 3D

| Paper | Venue | Link |
|-------|-------|------|
| Gaussian Grouping: Segment and Edit Anything in 3D Scenes | ECCV 2024 | [arXiv](https://arxiv.org/html/2312.00732v2) · [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04195.pdf) |
| Semantic NeRF: In-Place Scene Labelling with Implicit Representation | ICCV 2021 | [Project](https://shuaifengzhi.com/Semantic-NeRF/) · [CVF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhi_In-Place_Scene_Labelling_and_Understanding_With_Implicit_Scene_Representation_ICCV_2021_paper.pdf) |
| Only Semantic Information For NeRF Reconstruction | — | [arXiv](https://arxiv.org/html/2403.16043v1) |
| R5DGS: Semantic-Aware 4D Gaussian Splatting | OpenReview | [PDF](https://openreview.net/attachment?id=LFHg6tcn7e&name=pdf) |
| BEA-GS: Beyond Radiance Supervision for Precise Object Extraction | — | [arXiv](https://arxiv.org/html/2605.09662v1) |
| Space-Time Forecasting with Motion-aware Gaussian Grouping | CVPR 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lee_Space-Time_Forecasting_of_Dynamic_Scenes_with_Motion-aware_Gaussian_Grouping_CVPR_2026_paper.pdf) |
| Semantic Foam: Unifying Spatial and Semantic Scene Decomposition | CVPR 2026 | [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Sharafeldin_Semantic_Foam_Unifying_Spatial_and_Semantic_Scene_Decomposition_CVPR_2026_paper.pdf) |
| SLAG: Scalable Language-Augmented Gaussian Splatting | — | [arXiv](https://arxiv.org/html/2505.08124v2) |
| VoteSplat: Hough Voting Gaussian Splatting for 3D Scene Understanding | ICCV 2025 | [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_VoteSplat_Hough_Voting_Gaussian_Splatting_for_3D_Scene_Understanding_ICCV_2025_paper.pdf) |
| Beyond Averages: Open-Vocabulary 3D Scene Understanding with GS | — | [arXiv](https://arxiv.org/pdf/2509.12938) |
| Group Any Gaussians via 3D-aware Memory Bank | — | [arXiv](https://arxiv.org/html/2404.07977v3) |
| Gradient-Weighted Feature Back-Projection in 3DGS | — | [arXiv](https://arxiv.org/html/2411.15193v1) |
| Unsupervised Continual Semantic Adaptation Through Neural Rendering | CVPR 2023 | [CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Unsupervised_Continual_Semantic_Adaptation_Through_Neural_Rendering_CVPR_2023_paper.pdf) |
| Learning Semantic Fields of Dynamic Scenes from Monocular Videos | ICLR 2024 | [PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/801ec05b0aae9fcd2ef35c168bd538e0-Paper-Conference.pdf) |
| SG-NeRF: Neural Surface Reconstruction with Scene Graph Optimization | — | [arXiv](https://arxiv.org/html/2407.12667v1) |

### Structure-from-Motion & Camera Pose

| Paper | Venue | Link |
|-------|-------|------|
| GLUEMAP: Global SfM Meets Feedforward Reconstruction | CVPR 2026 | [Project](https://lpanaf.github.io/cvpr26_gluemap/) · [GitHub](https://github.com/colmap/gluemap) |
| DUSt3R: Geometric 3D Vision Made Easy | CVPR 2024 | [arXiv](https://arxiv.org/html/2312.14132v1) |
| MV-DUSt3R+: Single-Stage Scene Reconstruction in 2 Seconds | CVPR 2025 Oral | [Project](https://mv-dust3rp.github.io/) |
| COLMAP-Free 3D Gaussian Splatting | CVPR 2024 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Fu_COLMAP-Free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf) · [arXiv](https://arxiv.org/html/2312.07504v1) |
| CT-NeRF: Incremental Optimizing NeRF and Poses with Complex Trajectory | — | [arXiv](https://arxiv.org/html/2404.13896v2) |
| Fast Intrinsic–Extrinsic Calibration for Pose-Only SfM | Remote Sensing 2025 | [MDPI](https://www.mdpi.com/2072-4292/17/13/2247) |
| Assessing COLMAP, DROID-SLAM, and NeRF-SLAM in 3D Scene Reconstruction | Thesis | [PDF](https://lup.lub.lu.se/student-papers/record/9127302/file/9127313.pdf) |
| Object-Centric Pose Estimation by Scale Alignment of Ray Diffusion and ICP | Applied Sciences | [MDPI](https://www.mdpi.com/2076-3417/16/13/6624) |


<!--
================================================================
PROJECT NOTES — not rendered on GitHub
================================================================

## What We Already Have

- Shelf video sequences       ✅
- Extracted frame visuals     ✅  (product boxes visible)
- Trained metric model        ✅
- Labeling team               ✅

## Core Idea

Human labels bboxes on 1 frame → SAM converts to masks →
metric model assigns identity (no SKU names needed) →
Gaussian Grouping lifts labels into 3D →
scene renders from 10,000+ viewpoints with auto pixel-perfect masks →
YOLO trained on synthetic data.

## SKU Names — Are They Needed?

Not necessarily. Instance-level identity (cluster ID) is enough for:
  - bbox detection
  - OOS (empty shelf) detection

SKU names only required for per-product planogram compliance
(knowing *which* product should be in *which* slot).

## Metric Model Integration Options

A. Automatic identity labeler
   Embed bbox crops → cluster by similarity → assign cluster ID as
   "SKU proxy" for Gaussian Grouping. Removes need for humans to
   know or type SKU names.

B. Contrastive loss signal for Gaussian Grouping
   Replace manual SKU supervision with metric feature distances.
   Gaussians whose projected crops are similar in metric space
   get similar identity vectors.

C. "Detect then identify" at inference
   YOLO detects bboxes → metric model identifies product inside each box.
   Modular pipeline, independently trainable components.

Recommended: Human draws bbox → SAM → metric model → Gaussian Grouping →
             synthetic render → YOLO + metric model at inference.

## Pipeline Steps

Step 1 — Single-Frame Annotation
  - Draw bboxes on 1 reference frame per scene
  - SAM (bbox-prompted) → pixel-accurate instance masks
  - Metric model → cluster identity per crop
  - Export COCO JSON

Step 2 — Camera Pose Estimation (SfM)
  - GLUEMAP on existing sequences (robust to repeating shelf textures)
  - Alt: MV-DUSt3R+ for fast/edge use
  - Output: camera poses + sparse point cloud

Step 3 — Semantic 3DGS Training (Gaussian Grouping)
  - Initialize from SfM point cloud
  - Each Gaussian: 16-dim identity encoding vector
  - Loss: L_rgb + L_2D-identity (contrastive) + L_3D-KNN (spatial propagation)
  - ~15-30 min on A100/H100
  - Output: .ply with every product as addressable 3D object

Step 4 — Synthetic Data Rendering
  - Virtual cameras on random trajectories
  - Per pose: RGB frame + identity mask (Z-buffer correct occlusion)
  - Auto-derive 2D bboxes from masks
  - Augment: lighting (SH perturb), OOS (remove Gaussians),
    planogram violation (displace Gaussians)
  - Output: {rgb, mask, bbox COCO JSON, camera pose} x 10,000+

Step 5 — Downstream Training & Deployment
  - YOLO on synthetic + real-plus-synthetic mix
  - Up to +18% mAP vs real-only (literature)
  - INT8/FP16 quantization for edge
  - New product: label 1 frame → fine-tune → re-render → done in hours

## Progress

- [x] Sequences & frame visuals
- [ ] Step 1 — Single-frame annotation
- [ ] Step 2 — SfM / GLUEMAP
- [ ] Step 3 — Semantic 3DGS
- [ ] Step 4 — Synthetic rendering
- [ ] Step 5 — YOLO training & deployment

================================================================
-->
