---
author:
- |
  Edward Lee\
  `edwardnl@stanford.edu`
- |
  Saif Moolji\
  `smoolji@stanford.edu`
- |
  Umar Padela\
  `umarp@stanford.edu`
bibliography:
- references.bib
date: 2026-05-19
title: |
  CS 231N--- Project Proposal\
  **FocusFusion: Cross-Attention for LiDAR-Vision Segmentation**
---

# Proposal

3D semantic segmentation is the process of assigning a specific semantic
label to every individual point in a 3D LiDAR point cloud. It provides
the fine-grained scene understanding required for safe navigation and
complex decision-making. Despite its importance, 3D segmentation remains
a significant bottleneck due to the fundamental differences between 3D
data and 2D images. Unlike dense, ordered image grids, LiDAR point
clouds are inherently sparse and unstructured, consisting primarily of
empty space without a fixed spatial topology. While LiDAR provides
precise geometric and depth information, it lacks the rich textural and
semantic cues (such as color or signage) found in 2D camera images.
Solving this requires designing architectures that can effectively
navigate the "best of both worlds,\" creating perception systems that
are robust to sensor noise and capable of operating in real-time,
resource-constrained environments.\
We plan to use work from Waymo on 3D open-vocabulary panoptic
segmentation [@xiao20243dopenvocabularypanopticsegmentation] as
background for our architectural foundation. Specifically, we expand on
the paradigm of fusing a 3D LiDAR encorder with 2D vision features by
replacing projection-based fusion with a learned cross-attention
mechanism. Moreover, architectures like GAFusion
[@li2024gafusionadaptivefusinglidar] and DMFusion [@Yu2024DMFusion]
aggregate multi-frame LiDAR-camera features in Bird's Eye View (BEV) to
improve 3D object detection. Our work will extend this to denser
per-point 3D semantic segmentation.\
We are considering using autonomous vehicle data from either nuScenes
[@caesar2020nuscenesmultimodaldatasetautonomous] or Open Waymo
Perception [@sun2020scalabilityperceptionautonomousdriving],
particularly LiDAR and camera driving data along with semantic labels.

The proposed method is a Multi-Modal Attention-based Fusion framework
for 3D semantic segmentation that integrates LiDAR and vision \"expert\"
backbones through a dynamic weighting mechanism. Using frozen PTv3
[@wu2024pointtransformerv3simpler] and DINOv2
[@oquab2024dinov2learningrobustvisual] (or CLIP
[@radford2021learningtransferablevisualmodels]) backbones to extract
geometric and semantic features, the model then fuses these features via
a cross-attention layer. In this setup, LiDAR features serve as the
Query ($Q$) while visual patches act as the Key ($K$) and Value ($V$),
allowing the network to selectively attend to the most relevant visual
context for each 3D point. To enhance consistency, we are introducing a
Temporal Mechanism--implemented via either self-attention over a memory
bank of previous frames or a recurrent vector embedding--enabling the
model to resolve ambiguities in dynamic objects using historical
context.\
To evaluate the effectiveness of the fusion model, we will use the
following metrics like mean intersection over union (mIoU), mean
accuracy (mAcc), and frequency-weighted intersection over union (fwIoU),
which measure how well the predicted 3D segments align with the ground
truth labels. These three metrics together verify that our model is not
only accurate in a general sense but also reliable in detecting the
rare, high-stakes objects necessary for safe navigation.\
To qualitatively evaluate our model, we will prioritize attention map
visualizations, cross-modal error analysis, and robustness testing under
challenging conditions. From this, we can verify that the network is
identifying relevant visual cues, improving on the PTv3 baseline, and
performing well in edge cases. This ensures that our fusion strategy
remains reliable even when the sensor data quality is degraded, proving
the system's practical utility for real-world autonomous perception.

# References
