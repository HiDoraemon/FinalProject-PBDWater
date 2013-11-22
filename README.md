CIS 565 : Final Project
=====================
Harmony Li 
Joseph Tong

#Position Based Fluids

We want to implement a GPU based PBD fluid simulation stable enough to support 
large time steps for real-time applications. Through the enforcement of constant 
density positional constraints, the simulation will allow for incompressibility 
and convergence similar to smooth particle hydro-dynamic solvers. Our project 
will be largely based on Muller and Macklin’s paper “Position Based Fluids”.

-----

#Proposed Features:
*   Particle-based position based dynamics fluid simulator that runs on the GPU
*   Uses the density constraint to enforce incompressibility
*   Artificial pressure term to simulate surface tension
*   Vorticity confinement to replace energy
*   Viscosity term
*   GPU Hash Grid for optimization of finding particle neighbors
*   Meshless rendering (bilateral Gaussian or curvature flow)

-------

#Tentative Schedule

Mon, 11/25 (Alpha) | Setting up framework for constraint and particle system
 | Set up visualization for particles 
Mon, 12/9 (Last day of class) | begin writing paper
 | simulator with position based dynamics finished
 | GPU hash grid
Due date (Final) | meshless rendering finished 
 | paper finished

------

#External Libraries:
[Jacobi Iterator](https://code.google.com/p/jacobi-in-parallel/source/browse/jacobi/trunk/jacobiCUDA/cudaJacobi.cu?spec=svn34&r=34)

------

#References:
###Fluid Simulation
* [Position Based Fluids](http://mmacklin.com/pbf_sig_preprint.pdf) : M. Muller, M. Macklin 
* [Position Based Fluids : SIGGRAPH proceeding slides] (http://mmacklin.com/pbf_slides.pdf)
###Rendering
* [Screen Space Fluid Rendering with Curvature Flow] (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.909&rep=rep1&type=pdf) : W. van der Laan, S. Green, M. Sainz 
* [Reconstructing Surfaces of Particle-Based Fluids using Anisiotropic Kernels] (http://www.cc.gatech.edu/~turk/my_papers/sph_surfaces.pdf) : Yu, Turk


