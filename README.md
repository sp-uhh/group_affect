# Dynamics of Collective Group Affect: Group-level Annotations and the Multimodal Modeling of Convergence and Divergence

This repository contains the codebase for the paper:

Navin Raj Prabhu, Maria Tsfasman, Catharine Oertel, Timo Gerkmann, Nale Lehmann-Willenbrock, "Dynamics of Collective Group Affect: Group-level Annotations and the Multimodal Modeling of Convergence and Divergence", Under Review, Submitted to  IEEE Transactions on Affective Computing, Sep, 2024. [arxiv](https://arxiv.org/abs/2409.08578)


The repository contains the following:

## Data and analysis

1. [Data Reader and Classes](https://github.com/sp-uhh/group_affect/tree/main/readers)
2. [Annotation Analysis](https://github.com/sp-uhh/group_affect/tree/main/analysis/annotations)
3. [Inter-annotator agreement](https://github.com/sp-uhh/group_affect/blob/main/analysis/annotations/interannot_agreement.ipynb)
4. [Ground-truth consensus](https://github.com/sp-uhh/group_affect/blob/main/analysis/annotations/agreement_utils.py)


## Modeling

1. [Individual-level feature extractors](https://github.com/sp-uhh/group_affect/tree/main/feature_extractor)
   1. Audio (prosody)
   2. Video
2. [Dyad-level feature extractors](https://github.com/sp-uhh/group_affect/tree/main/groupsync/features/dyadic) ([Jobs](https://github.com/sp-uhh/group_affect/tree/main/jobs) for feature extraction available)
   1. Synchrony (correlation, lagged correlation)
   2. Convergence (global, symmetric and asymmetric)
2. [Convergence and Divergence analysis](https://github.com/sp-uhh/group_affect/tree/main/analysis/features)
3. [Group Affect recognition baselines](https://github.com/sp-uhh/group_affect/blob/main/models.py) (Use [configs](https://github.com/sp-uhh/group_affect/tree/main/configs) for baseline versions)
   1. Multi-layer Perceptron based model
   2. Graph Attention Network based model
