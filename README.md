# 3D Retail Analytics Research

A curated collection of papers and repositories for semantic 3D reconstruction, novel view synthesis, and synthetic data generation applied to retail shelf analytics.

---

## Papers and Repositories

## 0. Foundations of 3D Computer Vision & Projective Geometry

Before diving into advanced volumetric modeling, it is essential to establish a strong theoretical background in multiple view geometry, projective camera models, and the mechanics of differentiable rendering that make neural scene optimization possible.

| # | Resource | Key Focus |
|---|----------|-----------|
| 1 | [Stanford CS231A: Computer Vision, From 3D Reconstruction to Recognition](https://web.stanford.edu/class/cs231a/index.html) (Course) | **3D Basics:** Essential theoretical foundation covering camera models, projective geometry, and classical 3D reconstruction pipelines. |
| 2 | [Differentiable Rendering: A Survey](https://arxiv.org/abs/2006.12057) (CGF 2020) | **Optimization Core:** Explains how to bridge classical computer graphics and deep learning by backpropagating gradients through rendering processes. |
| 3 | [State of the Art on Neural Rendering](https://arxiv.org/abs/2004.03805) (EG 2020) | **Neural Rendering:** A comprehensive overview of neural rendering techniques prior to explicit 3DGS architectures. |
| 4 | [3D Representation Methods: A Survey](https://arxiv.org/html/2410.06475v1) | **3D Representations:** A comparative look at voxels, point clouds, meshes, and implicit functions to understand why the field evolved towards NeRF and 3DGS. |


## 1. Core 3DGS & NeRF

To dive into 3D scene reconstruction, one must first understand the foundations of volumetric modeling. This section covers the transition from implicit neural networks (NeRF) to the current real-time explicit standard, 3D Gaussian Splatting (3DGS).

| # | Resource | Key Focus |
|---|----------|-----------|
| 1 | [NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) (ECCV 2020) | The starting point of implicit neural 3D volumetric representation in the literature. |
| 2 | [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) (SIGGRAPH 2023) · [Repo](https://github.com/graphdeco-inria/gaussian-splatting) | The foundational paper for the current explicit, real-time 3DGS architecture that replaces NeRF. |
| 3 | [3DGS vs NeRF: Side-by-Side Reconstruction Comparison](https://assets-eu.researchsquare.com/files/rs-7300179/v1_covered_60fc3280-68ec-47c4-ae1d-228296ea12c4.pdf) | A comparative analysis of advantages and disadvantages for industrial transitions. |
| 4 | [VCR-GauS: View Consistent Depth-Normal Regularizer](https://proceedings.neurips.cc/paper_files/paper/2024/file/fc9f83d9925e6885e8f1ae1e17b3c44b-Paper-Conference.pdf) (NeurIPS 2024) | Essential regularization techniques for smooth surfaces and reflections, such as retail shelves. |
| 5 | [Nerfstudio: A Modular Framework for Neural Radiance Field Development](https://arxiv.org/abs/2302.04264) (SIGGRAPH 2023) · [Repo](https://github.com/nerfstudio-project/nerfstudio) | The most popular modular framework to train, test, and render NeRF and 3DGS models. |

## 2. Structure-from-Motion (SfM) & Camera Pose

Before a radiance field can be trained, one usually needs accurate camera poses. This section covers the transition from classical SfM pipelines (like COLMAP) to modern, deep learning-based, and feed-forward pose estimators that are revolutionizing 3D vision.

| # | Resource | Key Focus |
|---|----------|-----------|
| 1 | [Assessing COLMAP, DROID-SLAM, and NeRF-SLAM in 3D Scene Reconstruction](https://lup.lub.lu.se/student-papers/record/9127302/file/9127313.pdf) (Thesis) | **Baseline Setup:** Start here to understand the current landscape and limitations of classical and SLAM-based pose estimation. |
| 2 | [Fast Intrinsic–Extrinsic Calibration for Pose-Only SfM](https://www.mdpi.com/2072-4292/17/13/2247) (Remote Sensing 2025) | **Calibration Basics:** Understand the mechanics of intrinsic and extrinsic matrices before moving to deep learning methods. |
| 3 | [DUSt3R: Geometric 3D Vision Made Easy](https://arxiv.org/html/2312.14132v1) (CVPR 2024) | **The Paradigm Shift:** A fundamental breakthrough mapping unposed images directly to 3D point clouds without classical SfM. |
| 4 | [MV-DUSt3R+: Single-Stage Scene Reconstruction in 2 Seconds](https://mv-dust3rp.github.io/) (CVPR 2025 Oral) | **Evolution:** The immediate successor to DUSt3R, optimizing for multi-view consistency and extreme speed. |
| 5 | [COLMAP-Free 3D Gaussian Splatting](https://openaccess.thecvf.com/content/CVPR2024/papers/Fu_COLMAP-Free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf) (CVPR 2024) · [arXiv](https://arxiv.org/html/2312.07504v1) | **Integration:** Bridges the gap by showing how to train 3DGS *without* needing COLMAP preprocessing. |
| 6 | [GLUEMAP: Global SfM Meets Feedforward Reconstruction](https://lpanaf.github.io/cvpr26_gluemap/) (CVPR 2026) · [GitHub](https://github.com/colmap/gluemap) | **State-of-the-Art:** Merges the reliability of global SfM with the speed of feed-forward network priors. |
| 7 | [CT-NeRF: Incremental Optimizing NeRF and Poses with Complex Trajectory](https://arxiv.org/html/2404.13896v2) | **Dynamic Trajectories:** Explores joint optimization of poses and scene representation simultaneously. |
| 8 | [Object-Centric Pose Estimation by Scale Alignment of Ray Diffusion and ICP](https://www.mdpi.com/2076-3417/16/13/6624) (Applied Sciences) | **Specialized Application:** Applies pose estimation techniques specifically for precise, object-centric manipulation. |


## 3. Semantic Scene Understanding in 3D

Once geometric reconstruction is achieved, the next frontier is semantic understanding—segmenting, interpreting, and manipulating elements within the 3D space. The reading sequence moves from foundational NeRF concepts to advanced open-vocabulary 3DGS, and concludes with dynamic 4D scenes.

| # | Resource | Key Focus |
|---|----------|-----------|
| 1 | [Semantic NeRF: In-Place Scene Labelling with Implicit Representation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhi_In-Place_Scene_Labelling_and_Understanding_With_Implicit_Scene_Representation_ICCV_2021_paper.pdf) (ICCV 2021) · [Project](https://shuaifengzhi.com/Semantic-NeRF/) | **The Foundation:** The earliest core method for fusing 2D semantic labels into a 3D NeRF representation. |
| 2 | [Only Semantic Information For NeRF Reconstruction](https://arxiv.org/html/2403.16043v1) | **Ablation Study:** Explores what happens when geometry is reconstructed strictly from semantic cues. |
| 3 | [Gaussian Grouping: Segment and Edit Anything in 3D Scenes](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04195.pdf) (ECCV 2024) · [arXiv](https://arxiv.org/html/2312.00732v2) | **The 3DGS Leap:** The foundational paper for bringing 3D segmentation and editing into the 3DGS era (utilizing SAM). |
| 4 | [Group Any Gaussians via 3D-aware Memory Bank](https://arxiv.org/html/2404.07977v3) | **Extending Grouping:** Improves Gaussian grouping mechanisms for memory efficiency and 3D consistency. |
| 5 | [VoteSplat: Hough Voting Gaussian Splatting for 3D Scene Understanding](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_VoteSplat_Hough_Voting_Gaussian_Splatting_for_3D_Scene_Understanding_ICCV_2025_paper.pdf) (ICCV 2025) | **Methodological Innovation:** Uses traditional Hough voting concepts mapped to explicit 3D Gaussians for spatial queries. |
| 6 | [Beyond Averages: Open-Vocabulary 3D Scene Understanding with GS](https://arxiv.org/pdf/2509.12938) | **Open-Vocabulary (OV):** Transitions from fixed classes to understanding text-queried features in 3D. |
| 7 | [SLAG: Scalable Language-Augmented Gaussian Splatting](https://arxiv.org/html/2505.08124v2) | **Scalable OV:** Addresses the computational challenges of scaling open-vocabulary text embeddings to millions of Gaussians. |
| 8 | [Unsupervised Continual Semantic Adaptation Through Neural Rendering](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Unsupervised_Continual_Semantic_Adaptation_Through_Neural_Rendering_CVPR_2023_paper.pdf) (CVPR 2023) | **Adaptation:** Introduces continual learning without ground-truth 3D labels. |
| 9 | [Gradient-Weighted Feature Back-Projection in 3DGS](https://arxiv.org/html/2411.15193v1) | **Feature Engineering:** Deep dive into how 2D features map backwards into 3D Gaussians accurately. |
| 10 | [BEA-GS: Beyond Radiance Supervision for Precise Object Extraction](https://arxiv.org/html/2605.09662v1) | **Precise Extraction:** Moves beyond simple visual rendering into high-fidelity geometric extraction of semantics. |
| 11 | [Semantic Foam: Unifying Spatial and Semantic Scene Decomposition](https://openaccess.thecvf.com/content/CVPR2026/papers/Sharafeldin_Semantic_Foam_Unifying_Spatial_and_Semantic_Scene_Decomposition_CVPR_2026_paper.pdf) (CVPR 2026) | **Unified Theory:** A cutting-edge holistic approach treating space and meaning as a single structural representation. |
| 12 | [SG-NeRF: Neural Surface Reconstruction with Scene Graph Optimization](https://arxiv.org/html/2407.12667v1) | **High-Level Graphs:** Abstraction of scene semantics into relational node graphs. |
| 13 | [Learning Semantic Fields of Dynamic Scenes from Monocular Videos](https://proceedings.iclr.cc/paper_files/paper/2024/file/801ec05b0aae9fcd2ef35c168bd538e0-Paper-Conference.pdf) (ICLR 2024) | **Transition to 4D:** The bridge from static to dynamic (time-based) semantic fields. |
| 14 | [R5DGS: Semantic-Aware 4D Gaussian Splatting](https://openreview.net/attachment?id=LFHg6tcn7e&name=pdf) (OpenReview) | **Full 4D Semantics:** Integrates time, space, and meaning into moving Gaussian splats. |
| 15 | [Space-Time Forecasting with Motion-aware Gaussian Grouping](https://openaccess.thecvf.com/content/CVPR2026/papers/Lee_Space-Time_Forecasting_of_Dynamic_Scenes_with_Motion-aware_Gaussian_Grouping_CVPR_2026_paper.pdf) (CVPR 2026) | **Future Forecasting:** The current frontier—not just analyzing current 4D semantics, but predicting future scene states. |

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
