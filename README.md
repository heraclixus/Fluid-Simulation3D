# CFD_challenge

## Folder Structures of Repository 

- `config`: contains the `yaml` file for dataset, model, training, and visualization configurations. 
- `data`: contains the `npz` data file used for this challenge. 
- `docs`: TBD. Documentation for special choices of implementation, if needed.
- `logs`: generated log files from training and evaluation.
- `src`: the main entry to the source codes and the visualizations. 
    - `src/figs`: the generated visualization for training vs. test loss function + prediction scalar field vs. actual output scalar field. 
    - `src/layers`: custom layers used for `FNO4D` (implemented before I saw the acutal data).
    - `src/models`: different choices of models for the dataset (FNO-3D, FNO-4D, PINO)
    - `src/tests`: any unit tests, integration tests (if required)
    - `src/utils`: utility scripts. 
    - `main.py`, `train.py`: main entry points to the codebase. 


## Entry Point

The configurations can be modified by modifying the `config/FNO.yaml` at this moment. 

The entry point is simply: 
```
python main.py --model FNO --device cuda
python main.py --model dlResNet --device cuda
```

## Data 

The dataset is generated from 3-D Navier-Stokes equation in the domain $[0,1]^3 \subset \mathbb{R}^3$, with boundary conditions. The grid is coarse-grained but uniform, with $10\times 10\times10$ points in the 3D space. 

### Special Concern 1: 3D Modeling 
The original FNO paper deals with 2D Navier-Stokes equation and has two versions: 
- 2-D FNO, which does 2-D spectral convolution in space and time perform autoregressive/recurrent update in time.
- 3-D FNO, which does 3-D spectral convolution in the spacetime coordinate $(x,y,t)$. 

In our case, the data is generated in 3D:
- If the dataset has time steps then it would result in a __4-D FNO__. 
- However, it looks like the input and output pairs are simply just static 3D snapshots, so 3-D FNO can still be used. FNO can be unrolled for one time to generate the prediction. 

### Special Concern 2: Boundary Conditions 

The original FNO paper for the 2D Navier-Stokes equation is generated with periodic boundary condition. In our case, all 180 samples are generated with different boundary condition on the top wall and no-slip boundary condition on the rest 8 walls. 

- The 11th dimension of $x$ corresponds to the boundary conditions. This means in the deep learning task, this dimension should be considered. It can be regarded as a different spatial dimension.


### Special Concern 3: Low Resolution 

The data is collected at low resolution; it is possible that the validation/test dataset is at a higher resolution, making the original FNO not able to do well. 


### Special Concern 4: Turbulence 

FNO has seem reasonable performance on the 2D Navier-Stokes equation; 3-D situation is even more turbulent due to the additional degree of freedom. This is because the Reynold number scales with the scale. This challenge is related to the special conern 3, since its root cause the is the multiscale dynamics in the 3D Navier Stokes equation. 

### Special Concern 5: Static 

The kind of approach I am more familiar with is the dynamical system based approach, which relies on time series of moderate to long length, which makes it suitable to use ODE or transformer based models. However in this case we only have two static snapshots. 


## Models 

Upon first glance, a model to use is the neural operator family of models, since we don't know the underlying parametric PDE, and the operator learning framework is resolution-invariant, theoretically. The Fourier Neural Operator (FNO) is therefore the first attempt, used as a baseline model.

### Fourier-Neural Operator

First attempt: a FNO that:
- Performs spectral convolution in 4D: $(x,y,z,t)$. `models.FNO_4d.py` (_this was implemented before I saw the actual data_)
- Performs 3D spectral convolution then update temporal dimension autoregressively. `models.FNO_3d_time.py` 

### DlResNet

The main reference paper, [Learned Coarse Models for Efficient Turbulence Simulation](https://arxiv.org/abs/2112.15275), is a paper with scenario very similar to ours; it targets turbulence in 3D (Navier-Stokes) and can perform on coarse spatial and temporal scenarios. They use a Dilated ResNet Encode-Process-Decode architecture (__Dil-ResNet__) to perform the one-step prediction task, by predicting:
$\tilde{X}_{t+\Delta t} - \tilde{X}_t = \tilde{X}_t + NN(\tilde{X}_t; \theta)$
e.g., predicting the residual, with some generic convolution-based architecture. The __encode-process-decode__ architecture can have encoder and decoder with processing as described in their paper: 
- The encoder and decoder each consist of single linear convolutional layer.
- The processor consists of N = 4 dilated CNN blocks (dCNNn) connected in series with residual connections. 
- Each block consists of 7 dilated CNN layers with dilation rates of (1, 2, 4, 8, 4, 2, 1). 
- A dilation rate of N, indicates each pixel is convolved with pixels that are multiples of N pixels away 
    (N=1 reduces to regular convolution). 
- This model is not specialized for turbulence modeling per se, and its components are general-purpose tools for
    improving performance in CNNs. 
- Residual connections help avoid vanishing gradients, and dilations allow long-range communication 
    while preserving local structure. 
- All individual CNNs use a kernel size of 3 and have 48 output channels 
    (except the decoder, which has an output channel for each feature).
- Each individual CNN layer in the processor is immediately followed by a rectified linear unit (ReLU) activation function. 
    The Encoder CNN and the Decoder CNNs do not use activations.

### Factorized FNO

In the paper [Factorized Fourier Neural Operator](https://arxiv.org/abs/2111.13802), they claim to have a further improved version of FNO that achieves 83% improvement on the 2D Navier-Stokes equation. 


## Hyperparameter Tuning

The two models (FNO, dlResNet) has the following hyperparameters: 
- FNO:
    - `modes`: the number of Fourier modes for fft
    - `width`: the channel size in the spectral convolution layers. 
- dlResNet
    - `out_channels_lst`: the intermediate channel sizes for the conv3D layers. 
    - `kernel size`: the kernel size of the conv3d layers. 
    - `down_sample`: whether the spatial kernel and strides are used to reduce the spatial dimensions. 
In this case, we perform hyperparameter tuning using the RayTune package with PyTorch. 
