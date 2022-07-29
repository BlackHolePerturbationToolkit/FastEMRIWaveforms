from few.waveform import GenerateEMRIWaveform
wform = GenerateEMRIWaveform("Pn5AAKWaveform",use_gpu=False)
wform(1e5, 1e1, 0.2, 12.0, 0.4, 0.8, 1., 0.2, 0.4, 0.25, 0.45, 0., 0., 0., dt=15., T=2) # works fine
wform(1e5, 1e1, 0.2, 15.0, 0.4, 0.8, 1., 0.2, 0.4, 0.25, 0.45, 0., 0., 0., dt=15., T=2) # either GPUAssert or Segmentation fault