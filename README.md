## Neural Random Subspace (NRS), 3D Part, [Project Page](https://hansen7.github.io/NRS/)

This repo includes the content that we use to verify the effectiveness of NRS (Neural Random Subspace) module on 3D Point Cloud Recognition Task. It is based on the following implementations:

- https://github.com/WangYueFt/dgcnn
- https://github.com/yanx27/Pointnet_Pointnet2_pytorch

Baselines we use include [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) , [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) and [DGCNN](https://arxiv.org/abs/1801.07829) (optional)

We'd love you to cite our work if you find it is helpful :)



```
@misc{NRS-Yunhao,
    title={Neural Random Subspace},
    author={Yun-Hao Cao and Jianxin Wu and Hanchen Wang and Joan Lasenby},
    year={2019},
    eprint={1911.07845},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
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
