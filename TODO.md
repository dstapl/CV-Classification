# General
- [ ] Merge detection implementations of data simulations and analysis
  - [ ] Create third python file to act as a module

# Data Simulation
- [x] **URGENT** REDO iter_N = 1000 simulation for 2-2.5x period code
    - [x] Check if 2-2.5 period *or* 2-2.5h of observing



- [x] Include CCD binning
- [x] Adjust Observing time inputs
  - [x] exposure time limited to ~30s (Max 60s?)
  - [x] Set number of frames to maximum in the Observing period
    - [x] Reduce time sampling variation to mean~0, stddev based on pt5m specs (comparable to readout time?)
      - *Do not apply on first frame since t=0* (zero left pad list?)
    - [x] number of frames = (time period / exposure time) + sampling variation
      - Period used is either allowed Observing time, or superhump period
    - 
- [x] Set stddev of noise to size of dm (i.e. ~0.5-1)
- [x] Switch periodogram to Lomb-Scargle implementation
- [x] Improve p-value analysis (
  - (Choose *one* of these methods)
    - [x] 1
      - Remove 'regular' data generation
      - Calculate false-alarm spectral peak probability of each Lomb–Scargle periodogram for outburst data
      - 1-sample t-test (greater) against H0: 0
- [x] Create heatmap of p-values varying with m0 and dm

# Activity Parser
- [ ] Get J2000 coordinates of objects currently in outburst
  - [ ] Use SIMBAD? or other API using Astropy?