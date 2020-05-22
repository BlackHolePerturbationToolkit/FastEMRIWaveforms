# Slow waveform calculation

The code in this directory computes the waveform slowly. The code:
- densely samples the phase space trajectory and the waveform phases
- Interpolate the waveform amplitudes for each lmn-mode and computes the waveform by evaluating each of these and summing.

Compile the code using `scons`.