This code is used for the hybrid U-Net and Fourier neural operator (HUFNO) model developed in the paper: https://doi.org/10.1103/ymlb-wn4s (or https://arxiv.org/abs/2504.13126). The work is built upon the pioneering works by Li et al. (https://doi.org/10.48550/arXiv.2010.08895), Tran et al. (https://arxiv.org/abs/2111.13802), and Li et al. (https://doi.org/10.1063/5.0158830). 
The dataset for the 2d hill case at Re=700 and the trained model can be downloaded at https://www.kaggle.com/datasets/aifluid/2d-hill-re700.
The shape of the dataset is 21x400x32x33x16x3. 21 is the number of groups with the first 20 groups for training and the last 1 group for testing. 400 represents the number of time steps. 32x33x16 represents the shape of the uniform grids. 3 represents the flow variables u,v and w (note: p is not used for the current incompressible flow). In the dataset, the "npy" file contains the data, and the "pth" file is the model file which contains the weights of the model.
For further details and questions about this code, please contact: wangyp@sustech.edu.cn.


