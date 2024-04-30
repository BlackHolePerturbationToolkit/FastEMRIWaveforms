Paper skeleton and action items:

Trajectory: 
Check the trajectory for low eccentricities.
To do in terms of writing: 
0) check where we draw the line for the end of the trajectory, how far from the separatrix?
1) alternative parametrization (E,L) or (p,e) with and without phase integration
2) grid parametrization and tensor splines
To do in terms of plots and results: 
3) check ODE error variation: make a plot of the phase different as a function of the error for both rk4 and rk8
4) check trajectory against Scott, Zach and Soichiro
To do in terms of coding
5) try changing integrator to an implicit one: gsl_odeiv2_step_type *gsl_odeiv2_step_msadams

Amplitude: 
To do in terms of writing: 
1) write the new interpolation method bicubic+linear, and why NN failed, check against previous implementation
2) spherical harmonic basis
To do in terms of coding:
3) increase the dataset extend lmn boundary
To do in terms of plots and results: 
4) check SNR as a function of the parameter space, mode selection threshold, and radial index n (where to place the boundary in the radial index)

Waveform:
To do in terms of plots and results: 
1) accuracy, check against Scott, Zach, Soichiro waveforms
2) speed both in FD and TD
3) check SNR AAK compared to relativistic amplitudes
To do in terms of coding:
3) compare against the AAK, resolve the SSB frame
4) PE with mode selection and response, to investigate sky localization
5) redo mode selection threshold analysis as in previous paper

Minutes Apr. 29
- check of flux data with respect to the old one in Schwarzschild
- dividing the flux by the leading order is better than subtracting from Christian and Phil experience 
- calculate $\dot \omega / \omega^2$ and see where it is of order one to understand where to stop (Phil)
- regular grid or irregular timing (Christian)
- ask Scott how the convergence is done
- tricubic spline and filling memory experience from Zach https://github.com/znasipak/bhpwave/tree/tricubic
