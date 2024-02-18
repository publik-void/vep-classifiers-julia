# This is a tutorial meant to be run line by line e.g. in a Julia REPL or
# Jupyter notebook, so that the result of every step can be understood. It
# introduces the basic usage of the frame-wise and segment-wise machine learning
# models.

### Input data

# Let's simulate some CCVEP-type EEG, trigger, and labeling data. Let's say we
# have a stimulation pattern of 4 segments with 16 frames each. The four
# segments encode the bit string 0101 by alternating between normal phase
# (encoding a 0) and inverted phase (encoding a 1).
segment_labels = BitVector((false, true, false, true))

segments_per_trial = length(segment_labels)
frames_per_segment = 16

# The vector of lightness values (one per frame) would then look something like
# this:
stimulus_lightness_labels = [
  -.2, .6, -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -.6, .2,
  .2, -.6, 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., .6, -.2,
  -.2, .6, -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -.6, .2,
  .2, -.6, 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., .6, -.2]

frames_per_trial = length(stimulus_lightness_labels)

# Next, let's say we show the stimulus at a frame rate of 100Hz, while sampling
# the EEG and trigger data at 1000Hz, so that we have 10 samples per frame. The
# stimulus with its 64 frames would then have a duration of .64s.
samples_per_frame = 10
frames_per_second = 100
samples_per_second = samples_per_frame * frames_per_second
samples_per_segment = samples_per_frame * frames_per_segment

# Let's simulate an EEG signal that contains the VEP from this stimulus by
# adding the lightness labels to some uniform white noise. The EEG signal has 32
# channels and a duration of 2s, i.e. it is a 2000×32 matrix. The stimulus is
# shown after .5s. It shows up in the EEG with an additional delay of .1s. To
# resample the lightness vector from the frame rate to the EEG sampling rate, we
# will do a basic 0th-order interpolation using the `repeat` function.
n = 2000
m = 32
stimulus_start = 500
vep_delay = 100

stimulus_time_span = (stimulus_start - 1) .+
  (1 : frames_per_trial * samples_per_frame)
vep_time_span = stimulus_time_span .+ vep_delay

vep = repeat(stimulus_lightness_labels; inner = samples_per_frame)
eeg = rand(Float64, n, m) .* 2 .- 1
eeg[vep_time_span, :] .+= vep

# We won't do any pre-filtering for the sake of simplicity. Suffice it to say
# that it might be helpful to use highpass, lowpass, and notch filters on real
# EEG data to discard irrelevant frequency bands and power line interference.

# For the trigger signal, let's say we have two single-bit time series: The
# clock signal and the segment labeling. The clock signal in this case switches
# its value whenever a new stimulus frame begins and remains constant when no
# stimulus is shown. The segment labeling is only valid while the stimulus is
# shown and annotates each frame with a bit that indicates whether the phase of
# flickering is inverted.
clock = falses(n)
clock[stimulus_time_span] = repeat(BitVector((true, false));
  inner = samples_per_frame, outer = frames_per_trial ÷ 2)

segment_labeling = falses(n)
segment_labeling[stimulus_time_span] = repeat(segment_labels;
  inner = samples_per_segment)

### Using the frame-wise model

# We are going to use the model `FwSLT` to train a frame-wise classifier on our
# training dataset. We will then use the classifier to predict for each frame of
# our testing dataset if it was a dark or a bright frame. Here, the training and
# testing dataset will in both cases be our small, single-trial, simulated EEG
# and trigger data from above. Obviously, in reality, a larger dataset should be
# used, and it should be split into training and testing parts.

# We will have to load the module that contains the `FwSLT` model. We will also
# load the `VepModels` module, because it provides some utility functions for
# working with the EEG and trigger data. The `Thresholdings` module is needed
# for telling the model which type of thresholding to use.
using VepModels, VepModelFwSLT, Thresholdings

# We can start by defining the hyperparameters for the model. See the
# documentation of `FwSLT` for details.
fw = FwSLT(
  samples_per_segment / samples_per_second,
  (vep_delay - .5 * samples_per_segment) / samples_per_second,
  1.,
  :μ_σ,
  false,
  Binary,
  BitVector((false, true)),
  [.0],
  (atol = 1e-6, btol = 1e-6))

# To construct a feature matrix (`a`) and a label vector (`b`) from our input
# data, we will use the `ab` method for the `FwSLT` type. This method takes,
# among its other arguments, an object `label_markers` that indicates the time
# points in the signal where windows should be placed, and what label to
# associate with each of these time points. `label_markers` can assume many
# types – more info in the docs. Here, we will use a
# `Dictionaries.ArrayDictionary`, as it is well-suited for this case. We will
# also make use of the `VepModels.find_ticks` function to gather from the clock
# signal and into a vector the indexes of all time points where a new frame has
# started.
using Dictionaries
ticks = find_ticks(clock)
label_markers = ArrayDictionary(ticks, stimulus_lightness_labels)

# Each key in `label_markers` now represents the onset of a new frame and has
# the lightness of that frame associated with it as a label. Note that the
# labels don't necessarily have to be continuous lightness values. We could e.g.
# have made a separate trigger bit which only indicates whether a frame should
# be counted as black or white. We could then have mapped that to two discrete
# label values like e.g. `Int8(-1)` and `Int8(1)`. The classifier should still
# perform fine in that case.

# We can now create our feature matrix and label vector. The `ab` function takes
# as arguments the model hyperparameters, the EEG signal, the label markers,
# and the sampling rate. There are further optional arguments which provide
# opportunities for optimization in special cases – see the docs.
a, b = ab(fw, eeg, label_markers, samples_per_second)

# TODO: Continue by mentioning the `WindowMatrix` type, `mul_prepare`, FFT plan
# precomputation, and the warnings.
