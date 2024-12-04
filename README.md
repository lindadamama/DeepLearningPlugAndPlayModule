# DeepLearningPlugAndPlayModule

## 介绍

深度学习即插即用模块代码复现（代码 + 论文标题 + 论文地址）  基于PyTorch（参考论文给出的源代码）

代码输入张量 N×C×H×W，输出张量 N×C×H×W

创新点（涨点）必备！！！

缝合方式       串行、并行、组合、......（最好将不同模块组合成自己的新模块）

使用位置       特征提取层、任务后处理阶段、特征融合层、注意力模块、跳跃连接、编码器、解码器、...各种位置都可以，只要你讲的明白！！！



如果对你有帮助，点个Star鼓励！！

持续更新中......





## 已复现

|       模块       | 期刊/会议                                                    |                           论文标题                           | 论文地址                                                     |
|:--------------:|:---------------------------------------------------------| :----------------------------------------------------------: | ------------------------------------------------------------ |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|     SENet      | CVPR 2018                                                |               Squeeze-and-Excitation Networks                | https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html |
|      CBAM      | ECCV 2018                                                |          CBAM: Convolutional Block Attention Module          | https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html |
|    NonLocal    | CVPR 2018                                                |                  Non-Local Neural Networks                   | https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|     DANet      | CVPR 2019                                                |        Dual Attention Network for Scene Segmentation         | https://openaccess.thecvf.com/content_CVPR_2019/html/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.html |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|      ECA       | CVPR 2020                                                | ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks | https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|       CA       | CVPR 2021                                                |   Coordinate Attention for Efficient Mobile Network Design   | https://openaccess.thecvf.com/content/CVPR2021/html/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.html |
|     SA-Net     | ICASSP 2021                                              | SA-Net: Shuffle Attention for Deep Convolutional Neural Networks | https://ieeexplore.ieee.org/abstract/document/9414568        |
|     FcaNet     | ICCV 2021                                                |         FcaNet: Frequency Channel Attention Networks         | https://openaccess.thecvf.com/content/ICCV2021/html/Qin_FcaNet_Frequency_Channel_Attention_Networks_ICCV_2021_paper.html |
|      SRA       | ICCV 2021                                                |        Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction Without Convolutions                                                      |          https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Pyramid_Vision_Transformer_A_Versatile_Backbone_for_Dense_Prediction_Without_ICCV_2021_paper.html                                                                                                                |
|     SimAM      | PMLR 2021                                                | SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks | https://proceedings.mlr.press/v139/yang21o                   |
|      AFF       | WACV 2021                                                |                  Attentional Feature Fusion                  | https://openaccess.thecvf.com/content/WACV2021/html/Dai_Attentional_Feature_Fusion_WACV_2021_paper.html |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|      DAT       | CVPR 2022                                                |               Vision Transformer With Deformable Attention                                               |       https://openaccess.thecvf.com/content/CVPR2022/html/Xia_Vision_Transformer_With_Deformable_Attention_CVPR_2022_paper.html?ref=https://githubhelp.com                                                       |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|      MCA       | Engineering Applications of Artificial Intelligence 2023 | MCA: Multidimensional collaborative attention in deep convolutional neural networks for image recognition | https://www.sciencedirect.com/science/article/abs/pii/S0952197623012630 |
|      GLSA      | PRCV 2023                                                | DuAT: Dual-Aggregation Transformer Network for Medical Image Segmentation | https://link.springer.com/chapter/10.1007/978-981-99-8469-5_27?utm_source=chatgpt.com |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|                |                                                          |                                                              |                                                              |
|      MSPA      | Engineering Applications of Artificial Intelligence 2024 | Multi-scale spatial pyramid attention mechanism for image recognition: An effective approach | https://www.sciencedirect.com/science/article/abs/pii/S0952197624004196 |
|   CGAFusion    | TIP 2024                                                 |         DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution and Content-Guided Attention                                                     |                 https://ieeexplore.ieee.org/abstract/document/10411857                                             |
|                |                                                          |                                                              |                                                              |
|      MAB       | CVPR 2024                                                | Multi-scale Attention Network for Single Image Super-Resolution | https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Wang_Multi-scale_Attention_Network_for_Single_Image_Super-Resolution_CVPRW_2024_paper.html |
|      CGLU      |        CVPR 2024                                                  |       TransNeXt: Robust Foveal Visual Perception for Vision Transformers                                                          |                                 https://openaccess.thecvf.com/content/CVPR2024/html/Shi_TransNeXt_Robust_Foveal_Visual_Perception_for_Vision_Transformers_CVPR_2024_paper.html                                                                                                                      |
|   MetaFormer   | TPAMI 2024                                               |               MetaFormer Baselines for Vision                | https://ieeexplore.ieee.org/document/10304335                |
| AgentAttention | ECCV 2024                                                |        Agent Attention: On the Integration of Softmax and Linear Attention                                                      |        https://link.springer.com/chapter/10.1007/978-3-031-72973-7_8                                                      |
|      CSAM      | WACV 2024                                                |      CSAM: A 2.5D Cross-Slice Attention Module for Anisotropic Volumetric Medical Image Segmentation                                                        |         https://openaccess.thecvf.com/content/WACV2024/html/Hung_CSAM_A_2.5D_Cross-Slice_Attention_Module_for_Anisotropic_Volumetric_Medical_WACV_2024_paper.html                                                     |
|      AGF       | WACV 2024                                                |    MotionAGFormer: Enhancing 3D Human Pose Estimation With a Transformer-GCNFormer Network                                                          |      https://openaccess.thecvf.com/content/WACV2024/html/Mehraban_MotionAGFormer_Enhancing_3D_Human_Pose_Estimation_With_a_Transformer-GCNFormer_Network_WACV_2024_paper.html                                                        |





**即插即用卷积**

替换普通卷积即可有效涨点！

|      卷积模块      | 期刊/会议     |                           论文标题                           | 论文地址                                                     |
|:--------------:|:----------| :----------------------------------------------------------: | ------------------------------------------------------------ |
| MorphologyConv | TIP 2023  |           Single-Source Domain Expansion Network for Cross-Scene Hyperspectral Image Classification                                                                                                                                                                 |                https://ieeexplore.ieee.org/abstract/document/10050427                                                                                                                                                                                                                                                                                                        |
|     SCConv     | CVPR 2023 |      SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy                                                                                         |          https://openaccess.thecvf.com/content/CVPR2023/html/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.html                                                                                                                                                      |
|     DCNv4      | CVPR 2024 | Efficient Deformable ConvNets: Rethinking Dynamic and Sparse Operator for Vision Applications | https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_Efficient_Deformable_ConvNets_Rethinking_Dynamic_and_Sparse_Operator_for_Vision_CVPR_2024_paper.html |
|    StarConv    | CVPR 2024 |                Rewrite the Stars                                                                               |      https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Rewrite_the_Stars_CVPR_2024_paper.html                                                                                                                                                          |
|  DynamicConv   | CVPR 2024 |                   ParameterNet: Parameters Are All You Need for Large-scale Visual Pretraining of Mobile Networks                                                                                             |                                https://openaccess.thecvf.com/content/CVPR2024/html/Han_ParameterNet_Parameters_Are_All_You_Need_for_Large-scale_Visual_Pretraining_CVPR_2024_paper.html                                                                                                                                                                                                                             |
|     DEConv     | TIP 2024  |        DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution and Content-Guided Attention                                                      |                https://ieeexplore.ieee.org/abstract/document/10411857                                              |
|     WTConv     | ECCV 2024 |          Wavelet Convolutions for Large Receptive Fields                                                    |        https://link.springer.com/chapter/10.1007/978-3-031-72949-2_21                                                      |
|    CFBConv     | AAAI 2024 |                     SCTNet: Single-Branch CNN with Transformer Semantic Information for Real-Time Segmentation                                         |          https://ojs.aaai.org/index.php/AAAI/article/view/28457                                                    |
|                |           |                                                              |                                                              |
|                |           |                                                              |                                                              |
|                |           |                                                              |                                                              |

