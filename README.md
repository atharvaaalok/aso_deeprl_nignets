# Airfoil Shape Optimization (ASO) using DeepRL with NIGnets


## Getting Started
1. **Install xfoil-python** First we need to install
   [xfoil-python](https://github.com/DARcorporation/xfoil-python). This
   package allows us to call the xfoil code from python. Note that the big difference between this
   and other xfoil wrappers is that this does not call the executable rather the compiled code. So
   it is more memory efficient and is faster. But it can be tricky to get it installed.
   1. First make sure that you have a compiler for c/c++ and fortran (I have gcc).
   2. Now if you are on Mac (follow instructions from above package's README.md):
      1. git clone the above repository.
      2. go into the repository and run `pip3 install .`.
      3. If everything goes smoothly, run their example code from any directory on your system.
2. **NIGnets + xfoil-python**
   1. First, please have a look at `compute_L_by_D.py`. This is a python function that calls the
      code from `xfoil-python` package and returns the L-by-D ratio of an airfoil specified by its
      coordinates.
   2. Second, now please have a look at `fit_nignet_to_airfoil.py`. This script fits a 4 layer
      NIGnet to the NACA2412 airfoil. This is so that we can use that as a starting point. We can
      start from a symmetric airfoil and run DeepRL from there. I just fit 2412 as a test case to
      see if things work for cambered airfoils.
   3. Third, please have a look at `main.py`. This has the workflow we can use. Generate points, on
      our shape, then pass that to the `compute_L_by_D` function and get the L-by-D ratio.