# FUSE: Label-Free Image-Event Joint Monocular Depth Estimation via Frequency-Decoupled Alignment and Degradation-Robust Fusion

This repo is a PyTorch implementation of ***FUSE*** proposed in our paper: FUSE: Label-Free Image-Event Joint Monocular Depth Estimation via Frequency-Decoupled Alignment and Degradation-Robust Fusion

### Results and Weights

#### MVSEC outdoor_day11

|  Methods  | Input |    a1     |    a2     |    a3     |  Abs.Rel  |   RMSE    |  RMSElog  |
| :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|  E2Depth  |   E   |   0.567   |   0.772   |   0.876   |   0.346   |   8.564   |   0.421   |
| EReFormer |   E   |   0.664   |   0.831   |   0.923   |   0.271   |     -     |   0.333   |
|   HMNet   |   E   |   0.690   |   0.849   |   0.930   |   0.254   |   6.890   |   0.319   |
|  RAMNet   |  I+E  |   0.541   |   0.778   |   0.877   |   0.303   |   8.526   |   0.424   |
|  SRFNet   |  I+E  |   0.637   |   0.810   |   0.900   |   0.268   |   8.453   |   0.375   |
|   HMNet   |  I+E  |   0.717   |   0.868   |   0.940   |   0.230   |   6.922   |   0.310   |
|  PCDepth  |  I+E  |   0.712   |   0.867   |   0.941   |   0.228   |   6.526   |   0.301   |
|   Ours    |  I+E  | **0.745** | **0.892** | **0.957** | **0.196** | **6.004** | **0.270** |

#### MVSEC outdoor_night1

|  Methods  | Input |    a1     |    a2     |    a3     |  Abs.Rel  |   RMSE    |  RMSElog  |
| :-------: | :---: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|  E2Depth  |   E   |   0.408   |   0.615   |   0.754   |   0.591   |  11.210   |   0.646   |
| EReFormer |   E   |   0.547   |   0.753   |   0.881   |   0.317   |     -     |   0.415   |
|   HMNet   |   E   |   0.513   |   0.714   |   0.837   |   0.323   |   9.008   |   0.482   |
|  RAMNet   |  I+E  |   0.296   |   0.502   |   0.635   |   0.583   |  13.340   |   0.830   |
|  SRFNet   |  I+E  |   0.433   |   0.662   |   0.800   |   0.371   |  11.469   |   0.521   |
|   HMNet   |  I+E  |   0.497   |   0.661   |   0.784   |   0.349   |  10.818   |   0.543   |
|  PCDepth  |  I+E  | **0.632** |   0.822   |   0.922   |   0.271   |   6.715   |   0.354   |
|   Ours    |  I+E  |   0.629   | **0.824** | **0.923** | **0.261** | **6.587** | **0.351** |

#### Weights 

The metrics are the evaluation results under MVSEC outdoor_night1

|             |  a1   |  a2   |  a3   | Abs.Rel | RMSE  | RMSElog | Weights |
| :---------: | :---: | :---: | :---: | :-----: | :---: | :-----: | :-----: |
| Ours (vits) | 0.613 | 0.814 | 0.922 |  0.267  | 6.785 |  0.352  |         |
| Ours (vitb) | 0.632 | 0.827 | 0.925 |  0.270  | 6.445 |  0.348  |         |
| Ours (vitl) | 0.629 | 0.824 | 0.923 |  0.261  | 6.587 |  0.351  |         |

### Installation

```
pip install -r requirements.txt
```

### Data preparation

#### Data Download

* **MVSEC**: [Multi Vehicle Stereo Event Camera Dataset](https://daniilidis-group.github.io/mvsec/)
* **DENSE**: [Learning Monocular Dense Depth from Events](https://rpg.ifi.uzh.ch/E2DEPTH.html)
* **EventScape**: [Combining Events and Frames using Recurrent Asynchronous Multimodal Networks for Monocular Depth Prediction](https://rpg.ifi.uzh.ch/RAMNet.html)

Since the image, event, and depth labels in MVSEC are asynchronous, it is necessary to manually construct image-event-depth pairs, which can be achieved by running the script `scripts/process_mvsec_hdf5.py`

### Training

The training process is divided into three stages: *Feature Alignment*, *Feature Fusion*, and *Adaptation to the Target Dataset*. 

We use image-event pairs (without using depth ground truth) from EventScape and Depth Anything V2 to train the image-event joint encoder. We use DENSE and MVSEC as target datasets to verify the model effect. On DENSE and MVSEC, we only train the decoder.

**Feature Alignment**

Run the script `align_feature.sh`. 

You need to modify the variables `load_from` and `save_path` in the script, which represent the path of the pre-trained Depth Anything weights used and the path to save the training weights, respectively.

**Feature Fusion**

Run the script `fuse_feature.sh`. 

The variable `prompt_encoder_pretrained` in this script should be the event encoder weight path obtained in the feature alignment stage

**Adaptation to the Target Dataset**

Run the script `train.sh`. 

### Inference

Run the script `run.sh`. 

`load_from` indicates the path of the pre-trained weights to be loaded. 

### Evaluation

Run the script `eval.sh`. 

`predictions_dataset` and `target_dataset` represent the directory paths where prediction results and depth data are stored respectively. `clip_distance`indicates the maximum depth to be evaluated, which is 80 for MVSEC and 1000 for DENSE

## Acknowledgements
This project includes code from the following repositories:

- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) - We use Depth Anything V2 as the image depth foundation model