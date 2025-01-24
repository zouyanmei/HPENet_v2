# HPENet V2
Official PyTorch implementation for the following paper:

***Efficient Point Cloud Processing with High-Dimensional Positional Encoding and Non-Local MLPs***

*by Yanmei Zou, [Hongshan Yu](http://eeit.hnu.edu.cn/info/1289/4535.htm) *, Yaonan Wang, [Zhengeng Yang](https://gsy.hunnu.edu.cn/info/1071/3537.htm), Xieyuanli Chen, 	[Kailun Yang](https://yangkailun.com), [Naveed Akhtar](https://findanexpert.unimelb.edu.au/profile/1050019-naveed-akhtar)

## Features
In the project,:
1. We extend the unified ABS-REF view presented in [9] for efficient point cloud processing, where non-local MLPs are used to update non-local information, and HPE is deployed to effectively represent local geometric information. The ABS-REF paradigm underpins current high-performing point cloud modeling techniques, providing an intuitive framework for delineating the key strengths of these methods.

2. We propose a simple and effective Backward Fusion Module (BFM) to leverage contextual information. The BFM enables bilateral interaction between multi-resolution features.

3. Experiments on MLP-based methods are expanded considerably. We rethink various configurations of local aggregations in MLP-based methods and propose distinct aggregation strategies for different stages, departing from the consistent strategies employed across all stages in prior approaches[6 - 7].

4. To verify the effectiveness of the proposed modules, we introduce non-local MLPs, high-dimensional positional encoding, and backward fusion modules and incorporate them in the Transformer-based Point Transformer (PT) technique [13]. Our experiments demonstrate that these modules are highly compatible with PT, leading to significant improvements in its performance.
 
5. Through extensive evaluation, we achieve state-of-the-art (SOTA) results across all tasks studied, including 3D object classification, scene semantic segmentation, 3D object part segmentation while also being faster and more device-friendly than existing SOTA MLP-based methods.


## Installation

```
git clone git@github.com:zouyanmei/HPENet_v2.git
cd HPENet_v2
source install.sh
```
Note:  

1) the `install.sh` requires CUDA 11.1; if another version of CUDA is used,  `install.sh` has to be modified accordingly; check your CUDA version by: `nvcc --version` before using the bash file;
2) you might need to read the `install.rst` for a step-by-step installation if the bash file (`install.sh`) does not work for you by any chance;
3) for all experiments, we use wandb for online logging by default. Run `wandb --login` only at the first time in a new machine, or set `wandn.use_wandb=False` if you do not want to use wandb. Read the [official wandb documentation](https://docs.wandb.ai/quickstart) if needed.



## Usage 

**Check `README.md` file under `cfgs` directory for detailed training and evaluation on each benchmark.**  

For example, 
* Train and validate on ScanObjectNN for 3D object classification, check [`cfgs/scanobjectnn/README.md`](cfgs/scanobjectnn/README.md)
* Train and validate on ModelNet40 for 3D object classification, check [`cfgs/modelnet40ply2048/README.md`](cfgs/modelnet40ply2048/README.md)
* Train and validate on S3DIS for 3D segmentation, check [`cfgs/s3dis/README.md`](cfgs/s3dis/README.md)

Note:  
1. We use *yaml* to support training and validation using different models on different datasets. Just use `.yaml` file accordingly. For example, train on ScanObjectNN using HPENet V2: `CUDA_VISIBLE_DEVICES=0 bash script/main_classification.sh cfgs/scanobjectnn/hpenet_v2-s.yaml`, train on S3DIS using HPENet V2-xl: `CUDA_VISIBLE_DEVICES=1 bash script/main_segmentation.sh cfgs/s3dis/hpenet_v2-xl.yaml`.  
2. Check the default arguments of each .yaml file. You can overwrite them simply through the command line. E.g. overwrite the batch size, just appending `batch_size=32` or `--batch_size 32`.  


## Model Zoo

### ModelNet40 Classification


|       name           |   OA / mAcc    |                 Param.(M)                  | FLOPs(G)            |                 TP(Ins/Sec.)                
| :-------------------: | :----------------------------: | :-------------------------------------: | :----------------------------:| :---------------------------:
|		PointNeXt-S   | 93.2 ± 0.1 / 90.8 ± 0.2  |  1.4  | 1.6   | 4050
|		PointNeXt-S ($C_e$=64)     |  93.7 ± 0.3 / 90.9 ± 0.5 |  4.5 | 6.5   | 1952
|		HPENet      |     94.0 / 90.8      |  5.9     |  8.7   | 1308 
|		HPENet (SIN)      |   94.0 / 91.3      |  5.9     |    8.6  | 1206
|		HPENet V2-S    |   93.5 ± 0.2 / 90.8 ± 0.3      |    1.4   |    0.8   | 3859
|		HPENet V2-S*    |     93.7  / 90.8   |  1.4     |  0.8     | 3859 
|		HPENet V2-S ($C_e$=64)    |   93.7 ± 0.2  / 91.2 ± 0.4    |    5.1   |    3.1   | 2008
|		HPENet V2-S ($C_e$=64)*    |  94.0 / 91.7 | 5.1   |    3.1   | 2008 


### ScanObjectNN (Hardest variant) Classification


|       name           |   OA / mAcc    |                 Param.(M)                  | FLOPs(G)            |                 TP(Ins/Sec.)                
| :-------------------: | :----------------------------: | :-------------------------------------: | :----------------------------:| :---------------------------:
|		PointNeXt-S     | 87.7 ± 0.4  / 85.8 ± 0.6  | 1.4   | 1.6  | 4052
|		PointMetaBase-S   | 87.9 ± 0.2   / 86.2 ± 0.7  | 1.4 | 0.6 | 3140
|		PointMetaBase-S + X-3D | 88.8 / 87.2 | 1.5 | 0.6 | -
|		HPENet      | 88.9  / 87.6  |  1.7  | 2.2 | 2700
|		HPENet (SIN)      | 88.4  / 86.9  |  1.7  | 2.2 | 2455 
|		HPENet V2-S   | 88.4 ± 0.3      /   86.9 ± 0.4    | 1.5   | 0.8 | 3989
|		HPENet V2-S*  |  88.9  /  87.4     |    1.5   | 0.8 | 3989

### S3DIS (Area 5 and 6-fold) Segmentation

|       name             |    mIoU /OA /mAcc (S3DIS Area-5) |    mIoU /OA /mAcc (S3DIS 6-fold)    |    Param.(M)                  | FLOPs(G)            |                 TP(Ins/Sec.)             
| :--------------:       | :----------------------------: | :-------------------: | :----------------------------: | :-------------------------------------: | :----------------------------:
| 		Point Transformer	   | 70.4 / 90.8 / 76.5  | 73.5 / 90.2 / 81.9  | 7.8   | 5.6  | -
| 		Point Transformer V2       | 71.6 / 91.1 / 77.9  | - / - / -     | -     | - | -
| 		Point Transformer V3 | 73.4 / 91.7 / 78.9 | 77.7 / 91.5 / 85.3 | - | - | - 
| 		PointNeXt-S          | 63.4 ± 0.8 / 87.9 ± 0.3 / 70.0 ± 0.7   |   68.0 / 87.4 / 77.3  |0.8   | 3.6   | 410  
| 		PointNeXt-B          |67.3 ± 0.2 / 89.4 ± 0.1 / 73.7 ± 0.6 |  71.5 / 88.8 / 80.2     |3.8   | 8.9   | 293   
| 		PointNeXt-L        |69.0 ± 0.5 / 90.0 ± 0.1 / 75.3 ± 0.8    |  73.9 / 89.8 / 82.2	      | 7.1   | 15.2  | 221
| 		PointNeXt-XL         |70.5 ± 0.3 / 90.6 ± 0.2 / 76.8 ± 0.7 |   74.9 / 90.3 / 83.0  |41.6  | 84.8  | 98   
| 		PointMetaBase-L 		 | 69.5 ± 0.3 / 90.5 ± 0.1 / -     | 75.6 / 90.6 / -     | 2.7   | 2.0   | 245  
| 		PointMetaBase-XL		 | 71.1 ± 0.4 / 90.9 ± 0.1 / -     | 76.3 / 91.0 / -     | 15.3  | 9.2   | 130   
| 		PointMetaBase-XXL 		 | 71.3 ± 0.7 / 90.8 ± 0.6 / -     | 77.0 / 91.3 / -     | 19.7  | 11.0  | 114  
| 		PointMetaBase-L + X-3D  | 71.9 / 91.2 / - | 76.7 / 91.1 / - | 3.8 | 2.2 | 139 
| 		PointMetaBase-XL + X-3D | 72.1 / 91.4 / - | 77.7 / 91.6 / -  | 18.1 | 9.8 | 74
| 		HPENet			|    72.7 / 91.5 / 78.5     |    78.7 / 91.9 / 86.2   | 46.3  | 148.8| 57 
| 		HPENet (SIN)			|    72.4 / 91.0 / 78.9    |    78.2 / 91.7 / 86.1    |  46.3  | 147.2  |50
| 		[HPENet V2-S](https://drive.google.com/drive/u/0/folders/1eq795VdTS4woeOjUDTtnaM1kf0g1Miog) | 65.2 ± 0.3 / 89.1 ± 0.2 / 71.8 ± 0.4    |   70.5 / 88.7 / 79.6   |  0.7  |  2.6  | 323   	
| 		[HPENet V2-B](https://drive.google.com/drive/u/0/folders/1hibY6jZItPs_CjXA8NPN9sMBbT16OUgA) | 69.7 ± 0.3 / 90.4 ± 0.1 / 76.0 ± 0.4    |   76.0 / 91.0 / 84.3     | 1.8  |  3.7  | 260  
| 		[HPENet V2-L](https://drive.google.com/drive/u/0/folders/1m7Jp1LrlSY1E8dlPbgxYSB9MTBOh795e) | 71.0 ± 0.1 / 91.2 ± 0.2 / 77.0 ± 0.3     |  77.5 / 91.6 / 85.3     | 2.9 | 4.2   | 231 
| 		[HPENet V2-XL](https://drive.google.com/drive/u/0/folders/1pUWMBSRyF3EFcQsOTzapvMpt8zO0pA2T) | 72.3 ± 0.2 / 91.5 ± 0.1 / 78.4 ± 0.5  |  78.9 / 91.9 / 86.3     | 16.1  | 18.2  |  128 
| 		[HPENet V2-XL*](https://drive.google.com/drive/u/0/folders/1MYr-LiEvhBmksVRCTTAI_QZIhc8l6ou4) | 72.6 / 91.6 / 78.8   |   78.9 / 91.9 / 86.3    |  16.1  | 18.2  | 128 

### ScanNet V2 Segmentation

|       name             |    mIoU  |    Param.(M)                  | FLOPs(G)            |                 TP(Ins/Sec.)                       
| :--------------:       | :----------------------------: | :-------------------------------------: | :----------------------------:| :---------------------------:
| 		Point Transformer v2	| 75.4  | 12.8  | - |- 
| 		PointNeXt-S  | 64.5	| 0.7	| 4.7	| 357
| 		PointNeXt-B	| 68.4	| 3.8	| 9.1	| 289
| 		PointNeXt-L	| 69.4	| 7.1	| 15.4	| 219
| 		PointNeXt-XL	| 71.5  | 41.6  | 85.2  | 97
| 		PointMetaBase-L	| 71.0	| 2.7	| 2.1	| 242
| 		PointMetaBase-XL	| 71.8	| 15.3	| 9.6 	| 129
| 		PointMetaBase-XXL	| 72.8 |19.7  | 11.5 | 113
| 		PointMetaBase-L + X-3D	| 71.8 | 3.8 | 2.2 | 141
| 		PointMetaBase-XL + X-3D | 72.8 | 18.1 | 9.8 | 73
| 		HPENet	|  74.0 |  74.3  | 220.9 | 40 
| 		HPENet (SIN)	|  72.8 |  74.2  |  218.6 | 35
| 		[HPENet V2-S*](https://drive.google.com/drive/u/0/folders/1xROUhgnxwRf6YQjyLCqh6Dt4WBmHngnK)   |    66.3   |  0.8 | 3.3 |  302
| 		[HPENet V2-B*](https://drive.google.com/drive/u/0/folders/18_2OPKwVNfh9ejVWh4XJQhKIxCTaqLyT)   |    70.6   |  1.9 | 4.1 | 255
| 		[HPENet V2-L*](https://drive.google.com/drive/u/0/folders/1DFBFpeeQgVZB0Blg2cf6vNX2AOrSXyoW)	|   71.4    | 3.0 | 4.6 | 227
| 		[HPENet V2-XL*](https://drive.google.com/drive/u/0/folders/1Bl2ebaQu7jWGy-2wL8w36fo9TPuPHSEu)	|   73.3    |16.2 | 19.7| 125

### ShapeNetPart Segmentation

|       name             |   Cls. mIoU / Ins. mIoU    | Param. | FLOPs | TP            
| :--------------:       | :----------------------------:  | :--------------: | :--------------: | :--------------:
| 	HPENet			| 85.3 / 87.0     |   24.7     |  135.6 | 171 
| 	HPENet (SIN)				| 85.5 / 87.1  |    24.7   |  135.3 | 158
| 	[HPENet V2-S](https://drive.google.com/drive/u/0/folders/1T6qekhAFyIjPH3wcimHMjodzOkjwIUfa)										|   84.6 ± 0.1 / 86.8 ± 0.1 |    1.1   | 2.1 | 1567
| 	[HPENet V2-S(c=64)](https://drive.google.com/drive/u/0/folders/1InZjHtOJDgKXLYBxFS38tgLrcI4JEh6_) 										|  84.8 ± 0.4 / 86.8 ± 0.0  |  4.3     | 8.0 | 823
| 	[HPENet V2-S(c=160)](https://drive.google.com/drive/u/0/folders/1UjYJjsp-duqS4gQWYdvtQIIRbpekUCJN) 										|  85.4 ± 0.1 / 87.2 ± 0.1 |    26.3   | 48.9 | 293
| 	[HPENet V2-S*](https://drive.google.com/drive/u/0/folders/1D0qUW4mdsDCqclAz7d45M7Ah43Urjzjk)										|  85.4 / 87.3     |  26.3   | 48.9 | 293 








