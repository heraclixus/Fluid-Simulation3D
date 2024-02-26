# CFD_challenge


## Data 

The dataset is generated from 3-D Navier-Stokes equation in the domain $[0,1]^3 \subset \mathbb{R}^3$, with boundary conditions. The grid is coarse-grained, with $10\times 10\times10$ points in the 3D space. 

### Special Concern 1: 3D Modeling 


### Special Concern 2: Boundary Conditions 


### Special Concern 3: Turbulence 


### Special Concern 4: Coarse-Grained Grid 

## Models 

Upon first glance, a model to use is the neural operator family of models, since we don't know the underlying parametric PDE, and the operator learning framework is resolution-invariant, theoretically. Moreoever, The Fourier Neural Operator (FNO) is able to model 2D NS equation, and deal with the 3rd dimension as time. For a 3-D problem, an extension to FNO, called Geometry-informed Neural Operaotr (GINO) is able to deal with 3D geometry data, hence this is our first attempt. 

### Geometry-Informed Neural Operator 



### Physics-Informed Neural Operator 

Now that a baseline model is up and running for the 3-D problem, now we try to make it better by explicitly accounting for the boundary condition. This results in the Physics-Informed Neural Operator model (PINO)

### Turbulence 


## Visualizations 

