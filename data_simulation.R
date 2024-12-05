#' ---
#' title:  dm-dt detection simulation
#' author: Daniel Stapleton (dstapleton3@sheffield.ac.uk)
#' date:   2024-11-28
#' description: Want to find lowest dm that can be detected at 3\sigma significance level.
#' ---

#'@description
#'Simulate data of regular CV system and one exhibiting a superhump.
#'Magnitude change estimated from periodograms.
#'
#'@arguments
#'nframes 
#'period: Superhump period
#'
#'Exposure time (Ideally Nyquist sampling dt < period/2)
#'
#'Number of images taken
#'
#'Mean apparent magnitude of CV system
#'
#'STD Deviation in magnitude for random noise
#'
#'ditto time
#'
#'Amplitude of superhump magnitude oscillation

calc_dm <- function(nframes, dt, sigma_t, period, m0, sigma_m, dm) {
# Variation between captures (in seconds; TODO: Use Julian Time? from J2000)
  # (number of frames captured, exposure time, uncertanty on frame capture given by s.d, mean magnitude, error on m0 given by s.d., amplitude of super hump)
# 'sampling noise' = delay between sequential exposures & sigma_t = the error on the delay between exposures

sampling_noise <- rnorm(nframes, mean=0, sd=sigma_t)
    #rnorm creates a normal distribution.<- doesnt need to be included - just for generality 
  
# Generate time-series basis 
# 't' = time of observations - e.g., used as the x-axis for plots
t <- seq.int(1, dt*nframes, by=dt) + sampling_noise

# Generate noisy deviations to be added to both data sets
noise <- rnorm(nframes, mean = 0, sd = sigma_m)

#plot(t/60, noise, xlab="Time (mins)", ylab="Difference in Magnitude")
#title("Deviation in magnitude each frame")

# Now create data sets of CV system
# First regular (non-superhumping) CV
regular_data <- rep(m0, nframes) + noise
# Then superhumping CV
# Choosing approximate model
superhumping_data <- m0 + dm*sin(2*pi/period * t) + noise

#plot(t/360, superhumping_data, type="h", lty=2, col="red", xlab="Time (Hours)", ylab="Apparent Magnitude")
#lines(t/360, regular_data, col="black", lty=1)
#lines(t/360, rep(m0, length(t)),type="h", lty=3)
#legend(x="bottomright", legend = c("Superhump", "Regular"), lty=c(2,1),col = c("red", "black"))
#title("Simulated magnitude data")


# Estimate period from periodogram
# TODO: Tapering?
periodogram_regular <- spec.pgram(regular_data, plot=FALSE,taper=0, log="no")
est_freq_regular <- periodogram_regular$freq[periodogram_regular$spec == max(periodogram_regular$spec)]
est_period_regular <- round(dt/est_freq_regular,5)
est_dm_regular <- round(max(periodogram_regular$spec), 5)

periodogram_superhump <- spec.pgram(superhumping_data, plot=FALSE, taper=0, log="no")
est_freq_superhump  <- periodogram_superhump$freq[periodogram_superhump$spec == max(periodogram_superhump$spec)]
  #'find the point in the spectrum (freq est plt) where amplitude of frequencu est is max - spectrum of the F.T.'
est_period_superhump  <- round(dt/est_freq_superhump, 5)
  #calcs period from frequency 
est_dm_superhump <- round(max(periodogram_superhump$spec), 5)
  #cal
#plot(periodogram_superhump$freq, periodogram_superhump$spec, type="l",lty=2, col="red")
#lines(periodogram_regular$freq, periodogram_regular$spec, lty=1, col="black")
#legend(x="topright", legend = c(stringr::str_glue("Superhump: ", est_period_superhump), stringr::str_glue("Regular: ", est_period_regular)), lty=c(2,1),col = c("red", "black"))
#title("Periodogram")

  return(c(est_dm_regular, est_dm_superhump))
}


### USER INPUT

set.seed(Sys.time())
# Superhump period
period <- 50*60
# Exposure time (Ideally Nyquist sampling dt < period/2)
dt <- 90
# Number of images taken
nframes <- 0.5*60*60/dt # hours of observing time / time for 1 capture
# Mean apparent magnitude of CV system
m0 <- 15.0
# STD Deviation in magnitude for random noise
sigma_m <- 0.6
# ditto time
sigma_t <- 5
# Amplitude of superhump magnitude oscillation
dm <- 0.6





# Periodogram gives frequency and amplitude of frequency component
# Hypothesis test against null: (regular) and alternative (superhump) to 3sigma
  # creating a list of false alarm probablilities all data sets modeled 
dm_results <- data.frame()
for (i in 1:100){
  dm_results <- rbind(dm_results,calc_dm(nframes, dt, sigma_t, period, m0, sigma_m, dm))
}
#creating a list of false alarm probablilities all data sets modeled 
colnames(dm_results) <- c("regular","superhump")

# we run 100 simulation data sets with all the same parameters but random noise differaenciates them - see if the freqyency measured by periodogram is significant for that superhump amplitude of the 100 sets based on the mean false alarm probability

# Now do the hypothesis test
# running a T test to see if the mean false alarm probability 
t.test(dm_results$superhump, dm_results$regular, alternative="greater", paired=FALSE, mu=0)



