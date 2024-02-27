# Dataset Overview 

The dataset has length 180, corresponding to 180 different boundary conditions. Each boundary condition corresponds to the following triples 
- input: $(3,10,11,10)$
- output; $(3,10,10,10)$.
- meshgrid: $(3,10,10,10)$
where $3$ represents 3 spatial derivatives of $U$, and $(10,11,10)$ repesent the 3D grid coordinates.

The dataset is therefore a snapshot of the state at time = $3s$, a static image, as the input,  and then the output is the state at $3s+\Delta t$. 

If we concatenate the input and the output pair, then we have a snap shot of size $(2,3,10,10,10)$ where $2$ is the time dimension.


# Training and Test Setup 

The predictive task is to predict output given input. So in this case $X$ are the inputs and $Y$ are the outputs. The dataset can be indexed by the boundary condition on the top wall, giving the dataset a size of 180. Train-test-split is then performed on this dimension. 

For a standard Fourier Neural Operator setup, since the target value $y$ is unseen during training, it makes more sense, therefore, to perform either a 3-D convolution followed by auto-regressive one-step prediction. 