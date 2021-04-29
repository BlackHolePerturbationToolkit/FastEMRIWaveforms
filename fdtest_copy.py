from few.waveform import FastSchwarzschildEccentricFlux
import numpy as np

sum_kwargs = dict(pad_output=True, output_type="fd")

wave = FastSchwarzschildEccentricFlux(sum_kwargs=sum_kwargs)
wave(5e5,50,12,0.34,np.pi/3,np.pi/5)
