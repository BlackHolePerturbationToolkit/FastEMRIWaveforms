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
5) implement analytical or spline representation of elliptic integrals


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

Minutes for 07/05/24
@joshmat gave an update on his work towards generic PN+GSF kludge waveforms:

Given some set of frequencies in PN and GSF, examine their rates of change as a function of the frequencies
Assume they are linear combinations of one another and go from there; enforce that the frequencies match to leading order, which gets harder moving to generic systems
PN resummation: reasonably straightforward for quasicircular, harder in generic - include after generic is working
final output: kludge waveform models at 1PA, 2PA etc.
fits in easily with FEW: adapt the forcing function and evolve the frequencies. not sorted yet is mode amplitudes, 5PN is a starting point
Also some discussion on resonances...

@alvincjk : we would like to implement resonances properly in this waveform model as they are usually neglected despite their importance
@philip-lynch : jump condition approximation in resonances isn't accurate enough for a toy model, but might be marginally acceptable for the actual GSF problem. 1PA-level accuracy attainable with NIT, switching from orbit-averaged to orbital-timescale variations in integrator near resonances to smoothly model jump. however this takes O(10s) in Mathematica which will still to too expensive in C.
@niels : first project could be to efficiently find the resonance surfaces in parameter space. the difficulty of this changes based on parameter conventions used
Generally both technological and theoretical developments needed (mostly technological) in terms of an efficient resonance implementation. we need more efficient jumps through resonance and also a way to efficiently include them in the integrator.
@cchapmanbird : perhaps we can use the Julia DifferentialEquations.jl suite? has excellent event handling and integrator support, and can evaluate compiled C code very efficiently (no wrapping required).
Sufficient interest that we should probably have dedicated telecons for this. @alvincjk will send something round soon to organise this.

Action items:

@mikekatz04 : M1/M2/M3 install of FEW is still not working, it is essential that we fix this as soon as possible. Are you able to look into this?
@scott.a.hughes : question on mode index ranges for amplitudes. how were they chosen?
@niels will make a google drive where @cchapmanbird will upload a zip of the amplitude data, both raw and processed, along with the code to produce the processed grids.
Next meeting (fortnight):

@joshmat + @alvincjk : some work on efficient separatrix calculations
@joshmat : maybe some plots regarding the waveform work above
Feel free to add anything I have left out, or to correct anything I have wrongly summarised!

Minutes Monday May 13

compute Fisher matrix, mcmc agrees. Extension of the flux data to large p. Comparison AAK with Relativistic Kerr. Problem in the source frame of AAK and Relativistic Kerr. We need to get them fixed in the SSB frame. Ylm updated up to l=30. Horizon redshift for Circular Kerr. Caching. Second order self-force to be released, Phil looked at adiabaticity.


