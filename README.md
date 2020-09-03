## Neural Random Subspace (NRS), 3D Part, [Project Page](https://hansen7.github.io/NRS/)

This repo includes the content that we use to verify the effectiveness of NRS (Neural Random Subspace) module on 3D Point Cloud Recognition Task. 

![NRS-diagram](teaser/NRS-diagram.jpg)



It is based on the following implementations:

- https://github.com/WangYueFt/dgcnn
- https://github.com/yanx27/Pointnet_Pointnet2_pytorch

Major implementations of NRS can be found here: https://github.com/CupidJay/NRS_pytorch



For the details of ideas and results, please refer to our paper. We'd love you to cite it if you find it helpful :)

```bib
@article{NRS,
   title         = {Neural Random Subspace},
   author        = {Yun-Hao Cao and Jianxin Wu and Hanchen Wang and Joan Lasenby},
   year          = {2020},
   journal = {arXiv preprint arXiv:1911.07845}}
```



#### To start with:

##### - Data Preparation:

```bash
bash archive_bash/download_data.sh
```

##### - Training Models:

```bash
bash archive_bash/train_pointnet.sh
bash archive_bash/train_pointnet2.sh
bash archive_bash/train_dgcnn.sh
```

##### - FLOPS and \#Params

```bash
see utils/FLOPs_Calculator.py for details
```

##### - Inference Time

```bash
bash archive_bash/timer.sh
```



This version does not require to compile customized operators as in the original [PointNet++ repo](https://github.com/charlesq34/pointnet2)
