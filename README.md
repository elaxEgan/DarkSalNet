<div align="center">

<h1>DarkSalNet: Cross-Modal Fusion with UVI Enhancement for Salient Object Detection in Low-Light ORSI Images</h1>

<h3>✨under review by TCYB✨</h3>

<img src="https://github.com/elaxEgan/DarkSalNet/blob/main/assets/Visualization.png" width="100%"/>

## Abstract
> Salient object detection (SOD) in optical remote sensing images (ORSIs) remains challenging under low-light conditions, where reduced illumination impairs object visibility, contrast, and boundary clarity. Existing methods, primarily designed for well-lit scenes, suffer significant performance degradation in such scenarios. To address this, we propose DarkSalNet, a dual-branch framework tailored for low-light ORSISOD. It jointly exploits raw low-light inputs and UVI-enhanced features—derived from a perceptually guided color space transformation—to capture complementary global and structural cues. Two cross-modal integration modules are introduced: (1) a CrossModal Fusion (CMF) module that applies bidirectional gating for adaptive spatial calibration, and (2) a Semantic CrossAggregation (SCA) module that employs cross-attention and dual-path refinement to enhance semantic consistency. Extensive experiments on six synthetic low-light ORSI-SOD datasets and one real low-light NSI-SOD dataset demonstrate that DarkSalNet consistently outperforms state-of-the-art methods across varying illumination levels.

## Datasets
The synthetic dataset will be made available after the publication.

## Model evaluation

```
python inference.py
```

## Results

<div>
<img src="https://github.com/elaxEgan/DarkSalNet/blob/main/assets/result.jpeg" width="100%"/>
<div>
