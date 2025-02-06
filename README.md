# TCCL
# Twin Cross-scale Contrastive Learning with Multi-modality Fusion for Drug-Target Affinity Prediction

## Overview

<div align="center">
  <img src="model.png" alt="Model Architecture">
</div>
TCCL is an innovative contrastive learning framework with a twin cross-scale approach, designed to integrate multi-modal features for predicting drug-target binding affinity. A dual-stream encoder processes multi-modal data, extracting semantic and structural features of drugs and proteins at the molecular level, while balancing and synthesizing information from diverse sources to enhance feature representation. A network information aggregator extracts topological data from the drug-target bipartite graph, capturing interaction patterns at the network scale. Additionally, a twin cross-scale contrastive learning method, leveraging Semantic-Network Contrastive Learning (SENCL) and Structure-Network Contrastive Learning (STNCL), integrates multi-scale and multi-modal information. This approach enables efficient data fusion through molecular interaction understanding, improving the representation of drug and protein features and enhancing DTA prediction accuracy.


## Dependencies

```
- python                    3.7.16
- cuda                      11.7
- numpy                     1.21.5
- pandas                    1.3.5
- torch                     1.13.1+cu117
- rdkit                     2023.03.2
- torch_geometric           2.3.1
```

## Dataset

https://drive.google.com/drive/folders/1tKDDK0YA9poMdoKv-41QCCeKhvNVpMVP
## Training
```bash
# Running TCCL 
python inference.py 
```



