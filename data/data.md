The dataset has length 180, corresponding to 180 different boundary conditions. Each boundary condition corresponds to the following triples 
- input: $(3,10,11,10)$
- output; $(3,10,10,10)$.
- meshgrid: $(3,10,10,10)$
where $3$ represents 3 spatial derivatives of $U$, and $(10,11,10)$ repesent the 3D grid coordinates.

The dataset is therefore a snapshot of the state at time = $3s$, a static image, as the input,  and then the output is the state at $3s+\Delta t$. 