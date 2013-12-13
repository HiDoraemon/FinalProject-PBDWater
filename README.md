#CIS 565 : Final Project
##Position Based Fluids

Harmony Li, harmoli@seas.upenn.edu

Joseph Tong, josephto@seas.upenn.edu

-----

We want to implement a GPU based PBD fluid simulation stable enough to support 
large time steps for real-time applications. Through the enforcement of constant 
density positional constraints, the simulation will allow for incompressibility 
and convergence similar to smooth particle hydro-dynamic solvers. Our project 
will be largely based on Muller and Macklin’s paper “Position Based Fluids”.

-----

###Proposed Features:
*   Meshless rendering (bilateral Gaussian or curvature flow)
*   Caustics rendering for water?

###Implemented Features:
*   Uses the density constraint to enforce incompressibility
*   Artificial pressure term to simulate surface tension
*   Vorticity confinement to replace energy
*   Particle-based position based dynamics fluid simulator that runs on the GPU
*   GPU Hash Grid for optimization of finding particle neighbors

-------

###Tentative Schedule
* Mon, 11/25 ([Alpha](#alpha-results))
 + ~~Setting up framework for constraint and particle system~~
 + ~~Set up visualization for particles~~
* Mon, 12/9 (Last day of class)
 + ~~begin writing paper
 + ~~Simulator with position based dynamics finished
 + ~~GPU hash grid
* Due date (Final)
 + Meshless rendering finished 
 + ~~Paper finished

------

###Alpha Results

[Slides](https://github.com/harmoli/FinalProject-PBDWater/raw/master/CIS565-Alpha.pdf)

[Notes](https://github.com/harmoli/FinalProject-PBDWater/raw/master/CIS565-Alpha-Notes.pdf)

[Video](https://vimeo.com/80338399)

------

###Final Results

[Slides](https://github.com/harmoli/FinalProject-PBDWater/blob/master/CIS565-Final.pdf)

[Video](https://vimeo.com/)

[Final Paper](https://github.com/harmoli/FinalProject-PBDWater/blob/master/final_paper.pdf)

------

###Acknowledgments:

Many thanks to Mohammed Sakr(3dsakr@gmail.com), who has reached out with bug fixes and suggestions.

------

###References:
####Fluid Simulation
* [Position Based Fluids](http://mmacklin.com/pbf_sig_preprint.pdf) : M. Muller, M. Macklin 
* [Position Based Fluids : SIGGRAPH proceeding slides](http://mmacklin.com/pbf_slides.pdf)

####Rendering
* [Screen Space Fluid Rendering with Curvature Flow](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.909&rep=rep1&type=pdf) : W. van der Laan, S. Green, M. Sainz 
* [Reconstructing Surfaces of Particle-Based Fluids using Anisiotropic Kernels] (http://www.cc.gatech.edu/~turk/my_papers/sph_surfaces.pdf) : Yu, Turk


