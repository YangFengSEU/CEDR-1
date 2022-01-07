# Contrastive Embedding Distribution Refinement and Entropy-Aware Attention for 3D Point Cloud Classification

This repository is for Contrastive Embedding Distribution Refinement and Entropy-Aware Attention Network (CEDR) introduced in the following paper:  
**Contrastive Embedding Distribution Refinement and Entropy-Aware Attention for 3D Point Cloud Classification**  



## Network Architecture
![image](https://github.com/jinshuai224/CEDR/blob/master/img/CEDR.png)




## Updates

* **07/01/2021** Codes for both ```model.py``` and ```main.py``` are available now. 
* **05/01/2021** Update a pre-trained model (OA: **82.90%**, mAcc: **80.60%**) on ScanObjectNN via [google drive](https://drive.google.com/file/d/1R_yzyehbVDbOUVEj6Re-q0n1CZLE5MQw/view?usp=sharing).
* **05/01/2021** Pre-trained model (OA: **93.10%**, mAcc: **91.10%**) on ModelNet40 is available at [google drive](https://drive.google.com/file/d/1R_yzyehbVDbOUVEj6Re-q0n1CZLE5MQw/view?usp=sharing).



## Implementation Platforms

* Python 3.6
* [Pytorch](https://github.com/pytorch/pytorch) 0.4.0 with Cuda 9.1
* Higher Python/Pytorch/Cuda versions should also be compatible



## ModelNet40 Experiment 

**Test the pre-trained model:**

* download [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip), unzip and move ```modelnet40_ply_hdf5_2048``` folder to ```./data```

* put the pre-trained model under ```./checkpoints/modelnet```
* then run (more settings can be modified in ```main.py```):
```
python main.py --exp_name=gbnet_modelnet40_eval --model=gbnet --dataset=modelnet40 --eval=True --model_path=checkpoints/modelnet/gbnet_modelnet40.t7
```



## ScanObjectNN Experiment 

**Test the pre-trained model:**

* download [ScanObjectNN](https://github.com/hkust-vgd/scanobjectnn/), and extract both ```training_objectdataset_augmentedrot_scale75.h5``` and ```test_objectdataset_augmentedrot_scale75.h5``` files to ```./data```
* put the pre-trained model under ```./checkpoints/gbnet_scanobjectnn```
* then run (more settings can be modified in ```main.py```):
```
python main.py --exp_name=gbnet_scanobjectnn_eval --model=gbnet --dataset=ScanObjectNN --eval=True --model_path=checkpoints/gbnet_scanobjectnn/gbnet_scanobjectnn.t7
```



## Pre-trained Models

* Python 3.6, Pytorch 0.4.0, Cuda 9.1
* 8 GeForce RTX 2080Ti GPUs
* using default training settings as in ```main.py```

| Model            | Dataset             |#Points             | Data<br />Augmentation | Loss | Performance<br />on Test Set            | Download<br />Link   |
|:----------------:|:-------------------:|:-------------------:|:----------:|:-----------------:|:-------------------------------------------------------------------------------:|:------:|
| GBNet | ModelNet40 | 1024 | random scaling<br />and translation | cross-entropy<br />with label smoothing                 | overall accuracy: **93.10%**<br />average class accuracy: **91.10%**                                 | [google drive](https://drive.google.com/file/d/1R_yzyehbVDbOUVEj6Re-q0n1CZLE5MQw/view?usp=sharing) |
| GBNet | ScanObjectNN | 1024 | random scaling<br />and translation | cross-entropy<br />with label smoothing                 | overall accuracy: **82.90%**<br />average class accuracy: **80.60%**                                   | [google drive](https://drive.google.com/file/d/1R_yzyehbVDbOUVEj6Re-q0n1CZLE5MQw/view?usp=sharing) |



## Acknowledgement

The code is built on [GBNet](https://github.com/ShiQiu0419/GBNet). We thank the authors for sharing the codes. We also thank the Big Data Center of Southeast University for providing the facility support on the numerical calculations in this paper.

