# Data Simulation
- [ ] **URGENT** REDO iter_N = 1000 simulation for 2-2.5x period code
    - [ ] Check if 2-2.5 period *or* 2-2.5h of observing



- [ ] Include CCD binning
- [ ] Adjust Observing time inputs
  - [ ] exposure time limited to ~30s (Max 60s?)
  - [ ] Set number of frames to maximum in the Observing period
    - [ ] Reduce time sampling variation to mean~0, stddev based on pt5m specs (comparable to readout time?)
      - *Do not apply on first frame since t=0* (zero left pad list?)
    - [ ] number of frames = (time period / exposure time) + sampling variation
      - Period used is either allowed Observing time, or superhump period
    - 
- [ ] Set stddev of noise to size of dm (i.e. ~0.5-1)
- [ ] Switch periodogram to Lomb-Scargle implementation
- [ ] Improve p-value analysis (
  - (Choose *one* of these methods)
    - [ ] 1
      - Remove 'regular' data generation
      - Calculate false-alarm spectral peak probability of each Lomb–Scargle periodogram for outburst data
      - 1-sample t-test (greater) against H0: 0
    - [ ] 2
      - Keep 'regular' data
      - Calc false-alarm probability for both generated data sets (regular,superhump)
      - Accumulate probabilities in 2 column data frame
      - 2-sample t-test (greater) x=superhump, y(null)=regular
- [ ] Create heatmap of p-values varying with m0 and dm
