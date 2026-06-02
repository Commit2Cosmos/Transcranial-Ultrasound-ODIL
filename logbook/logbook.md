# Logbook

## 27/05/2026: Weekly meeting summary

The next meeting will be on the 8th June.

Before the next meeting, the aim is for each of us to develop our own code (working collaboratively) to reproduce Fig 1 of the ODIL paper. The focus of this figure is comparing the solution of the 1D wave equation using a PINN and ODIL. It solves solely the forward problem. It also includes an ablation study which compares the error, training epochs, and execution time for both solutions. Some key details include:

- ODIL solution was found on a 25 x 25 grid.
- PINN consisted of two hidden layers with 25 neurons each.
-  For a given number of parameters N, ODIL represents the solution on a $\sqrt{N} \times \sqrt{N}$ grid, while PINN consists of two equally sized hidden layers with tanh activation.
- Collocation points are specific locations in the domain of a differential equation where the approximate solution is required to satisfy the equation exactly.
-  The number of collocation points for PINN is fixed and amounts to 8192 points inside the domain and 768 points for the initial and boundary conditions functions.
- The execution time is a product of the number of optimization epochs required to reach a 150% of the error obtained after 80000 epochs and an average execution time over the last 100 epochs.
- The ablation study featured a comparison of training with L-BFGS-B and Newton's method. The error metric was the $L2$ norm.
- In the figure, the plots compare against a reference solution computed using finite differences.

General order of affairs:

- Obtain reference solution
- Implement ODIL
- Implement PINN
- Experiment

Other notes fromt today:

- The end goal of this exercise is to understand the ODIl methodology in practice. Experiements should explore its limitations, et cetera. This can also be a guide for estimating computational requirements and time to solution.
- The hope is that ODIL will bypass the requirement for AWI.
- You can spawn a notebook on the RCS using [JupyterHub](https://jupyter.cx3.rcs.ic.ac.uk/hub/login?next=%2Fhub%2Fspawn).
- Consider setting up Weights & Biases to track experiments over time

## 28/05: progress update

Regarding the above task:

- Created `notebooks/reproduction/{reference.ipynb,odil.ipynb,pinn.ipynb}`
- Recreated the reference solution. Upon looking at the ODIL codebase, realised the initial condition the paper states in inconsistent with the plot. Mirrored their implementation as a result.
- Recreated the ODIL result. 
    - Discretised the wave equation as specified in the paper along with boundary conditions
    - Split functionality into a `Wavefield` class and a `DiscretePDELoss` class. The later has a `__call__` method which is used by `scipy.optimize` to evaluate the full functional
    - Implemented residual tracking and corresponding plotting functionality.
    - Ran, timed, and plotted results for ODIL for L-BFGS-B and Newton-CG with wavefield initialised with zeros and random values.

Next steps: 
- [ ] Implement the PINN approach
- [ ] Combine with reference and ODIL results in single plot including trace samples
- [ ] Investigate limitations of ODIL along with proper runtime statistics

## 29/05: progress update

I have implemented the PINN approach, but I am struggling to match their results. A key issue is the network is failing to replicate the high frequency information contained in the reference solution. I originally trained with just Adam and it produced a decent result but stagnated after ~10000 epochs. Now I am playing around with switching to L-BFGS later in the training to try to avoid getting stuck in local minima. I fear the network they said they used just doesn't have enough capacity to represent the high frequency data, though. The mismatch between what they stated and what they implemented in the code base for the reference solution is making me wary. They didn't mention anything about training strategy, and the architecture is slightly hazy, hence the difficulty. Will perservere.

## 01/06: Weekly meeting

Brief discussion of progress on the initial task. I have now got the ODIL and PINN solution working.

Also discussed how we could specialise. They suggested 3 options:

1. Software/ code optimisation focus: generic 2D/ 3D ODIL solver for forward problems in wave modelling
2. Domain focus: ODIL for generating accurate skull models as a preconditioner for FWI
3. Inverse focus: 2D FWI using ODIL

I also brought up the possibility of an UQ study, and they seemed interested, so maybe worth reading about. If that does not go ahead, opt. 1 and 3 seem interesting to me, but opt. 1 is probably less likely due to the need for GPU programming, etc.

## 02/06: progress update

I have finished the baseline ODIL and PINN figure reproduction. Next, I will either:

- Complete a similar ablation study
- Extend the ODIL framework to invert for a non-constant wavespeed

Note that the latter is effectively 1D FWI. In practice, we will need to add a data residual to the functional that measures the difference between the target and observed data at discrete point(s).
