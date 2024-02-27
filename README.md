# CFD_challenge


## Data 

The dataset is generated from 3-D Navier-Stokes equation in the domain $[0,1]^3 \subset \mathbb{R}^3$, with boundary conditions. The grid is coarse-grained but uniform, with $10\times 10\times10$ points in the 3D space. 

### Special Concern 1: 3D Modeling 
The original FNO paper deals with 2D Navier-Stokes equation and has two versions: 
- 2-D FNO, which does 2-D spectral convolution in space and time perform autoregressive/recurrent update in time.
- 3-D FNO, which does 3-D spectral convolution in the spacetime coordinate $(x,y,t)$. 

In our case, the data is generated in 3D. Hence giving rise to the 4-D FNO as a first baseline $(x,y,z,t)$. 

### Special Concern 2: Boundary Conditions 

The original FNO paper for the 2D Navier-Stokes equation is generated with periodic boundary condition. In our case, all 180 samples are generated with different boundary condition on the top wall and no-slip boundary condition on the rest 8 walls. 

### Special Concern 3: Low Resolution 

The data is collected at low resolution; it is possible that the validation/test dataset is at a higher resolution, making the original FNO not able to do well. 


### Special Concern 4: Turbulence 

FNO has seem reasonable performance on the 2D Navier-Stokes equation; 3-D situation is even more turbulent due to the additional degree of freedom. This is because the Reynold number scales with the scale. This challenge is related to the special conern 3, since its root cause the is the multiscale dynamics in the 3D Navier Stokes equation. 


## Models 

Upon first glance, a model to use is the neural operator family of models, since we don't know the underlying parametric PDE, and the operator learning framework is resolution-invariant, theoretically. The Fourier Neural Operator (FNO) is therefore the first attempt, used as a baseline model.

### Fourier-Neural Operator

First attempt: a FNO that:
- Performs spectral convolution in 4D: $(x,y,z,t)$. `models.FNO_4d.py` 
- Performs 3D spectral convolution then update temporal dimension autoregressively. `models.FNO_3d_time.py` 

### Physics-Informed Neural Operator 

Now that a baseline model is up and running for the 3-D problem, now we try to make it better by explicitly accounting for the boundary condition. This results in the Physics-Informed Neural Operator model (PINO). The key is to add physics-informed loss based on Boundary condition and the 3D Navier-Stokes equation. 

From the PINO paper: neural operators cannot perfectly approximate the ground-truth operator when only coarse-resolution training data is provided. 



### Turbulence 
TODO 

## Visualizations 
TODO 
