# This is a tutorial meant to be run line by line e.g. in a Julia REPL or
# Jupyter notebook, so that the result of every step can be inspected and
# understood. It introduces the basic usage of the frame-wise and segment-wise
# machine learning models.

import Random
Random.seed!(0)


### Input data

# Let's simulate some CCVEP-type EEG, trigger, and labeling data. Let's say we
# have a trial of 192 segments with 16 frames each. The 192 segments encode the
# bit string 01010… of length 192 by alternating between normal phase (encoding
# a 0) and inverted phase (encoding a 1).
segments_per_trial = 192
segment_labels = repeat(BitVector((false, true)), segments_per_trial ÷ 2)

# The reason we use a long and monotonous stimulus in this contrived example is
# so we can construct it easily by just concatenating a bunch of single segments
# with fades on both sides. These would look something like this.
segment_lightness_labels_normal_phase =
  [.2, -.6, 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., .6, -.2]
segment_lightness_labels_inverted_phase =
  -segment_lightness_labels_normal_phase
frames_per_segment = length(segment_lightness_labels_normal_phase)

# And the whole stimulus as a vector of lightness values (one per frame):
stimulus_lightness_labels = vcat((bit
  ? segment_lightness_labels_inverted_phase
  : segment_lightness_labels_normal_phase
  for bit in segment_labels)...)

frames_per_trial = length(stimulus_lightness_labels)

# Next, let's say we show the stimulus at a frame rate of 100Hz, while sampling
# the EEG and trigger data at 1000Hz, so that we have 10 samples per frame. The
# stimulus with its 30720 frames would then have a duration of 30.72s. Yes, this
# is one unrealistically long stimulus. Normally, we would have multiple shorter
# stimuli.
samples_per_frame = 10
frames_per_second = 100
samples_per_second = samples_per_frame * frames_per_second
samples_per_segment = samples_per_frame * frames_per_segment

# Let's simulate an EEG signal that contains the VEP from this stimulus by
# adding the lightness labels to some uniform white noise. The EEG signal has 8
# channels and a duration of 40s, i.e. it is a 40000×8 matrix. The stimulus is
# shown after 1.5s. It shows up in the EEG with an additional delay of .1s. To
# resample the lightness vector from the frame rate to the EEG sampling rate, we
# will do a basic 0th-order interpolation using the `repeat` function.
n = 40000
m = 8
stimulus_start = 1500
vep_delay = 100
noise_amplitude = 2^5

stimulus_time_span = (stimulus_start - 1) .+
  (1 : frames_per_trial * samples_per_frame)
vep_time_span = stimulus_time_span .+ vep_delay

vep = repeat(stimulus_lightness_labels; inner = samples_per_frame)
eeg = (rand(Float64, n, m) .* 2 .- 1) .* noise_amplitude
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


## Preparing the model inputs

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

# The `FwSLT` model implements the frame-wise sliding window classifier as
# described in my Master's Thesis (and very similar to what Martin Spüler,
# Sebastian Nagel, Alexander Blöck, and probably others have been doing prior to
# that). It classifies the lightness value of individual time points by placing
# a window around them in the EEG signal and running three stages of machine
# learning on them. The first stage is a feature standardization by affine
# (linear) transformation. The `:μ_σ` above specifies that the features should
# be standardized to have a mean of zero and a variance of 1. The second stage
# is a ridge regression. With the above parameters, the ridge regression will
# have a λ parameter of 1 and no bias term. I have usually not seen performance
# improvements from incorporating a bias term in the feature matrix, though I
# suspect that would change when the EEG siganl is not highpass-filtered or when
# the label vector has a mean that is significantly different from zero.
# The third stage performs a thresholding to convert continuous lightness values
# to discrete labels such as "black" and "white" (represented by `false` and
# `true` in this case). The `atol` and `btol` are stopping criteria for the
# optimization algorithm. Using 64-bit floating point precision and the
# tolerance value of `1e-6` has usually resulted in a good trade-off between
# accuracy of the solution and runtime for me. It is a bit unfortunate, but I
# would not recommend using 32-bit floats for the EEG signal / feature matrix.

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
# perform fine in that case. Predictions of the classifier will be discrete in
# any case due to the thresholding.

# We can now create our feature matrix and label vector. The `ab` function takes
# as arguments the model hyperparameters, the EEG signal, the label markers,
# and the sampling rate. There are further optional arguments which provide
# opportunities for optimization in special cases – see the docs.
a, b = ab(fw, eeg, label_markers, samples_per_second)

# When evaluating `a`, we can see in the console output that its type is a
# `WindowArrays.WindowMatrix`. This is a custom matrix type that internally only
# stores a vector of data (`a.v`), a vector of indexes into the data vector
# (`a.i`), and the number of matrix columns (`a.k`). The matrix has one row for
# every element in `a.i`, which is a view of length `a.k` into `a.v`.

# The `ab` method flattens the `eeg` signal matrix into a vector such that each
# channel is enumerated for the first time point, then each channel for the
# second time point, and so on. This is why the number of columns of `a` is the
# window size in samples multiplied by the number of channels. Each row is a
# window into the flattened signal matrix where the first `m` values represent
# the first time point included in the window, the following `m` values
# represent the next time point, etc. To access only the channel `j` from a row
# `i`, one would use something like `a[i, j : m : end]` or
# `view(a, i, j : m : lastindex(a, 2))`.

# In the case of overlapping windows, the `WindowMatrix` is more space-efficient
# than a standard matrix that stores each of its elements individually (and
# hence redundantly). It also allows for a faster matrix multiplication that
# makes use of the fast Fourier transform (FFT) algorithm. Some steps of the
# fast matrix multiplication can be precomputed to be reused in all subsequent
# multiplications. This is done with the `WindowArrays.mul_prepare` function.
# The `WindowMatrix` object has two more fields in addition to those mentioned
# above: `a.mp` and `a.mpa`, which hold the precomputed data for multiplications
# of the form `a * x` and `a' * x`, respectively. Before calling `mul_prepare`,
# those `a.mp` and `a.mpa` will both be `nothing`. We will not call
# `mull_prepare` manualy here, because it will be done inside the `fit` method
# that we will invoke further below, but the call would look something like
# this:
#using WindowArrays: mul_prepare
#a = mul_prepare(a)

# The FFT-based multiplication makes use of the FFTW library. The FFTW library
# is designed in such a way that an FFT algorithm must first be planned before
# being executed. The reason for this is that there are many ways to implement
# the same FFT algorithm, and some of those will run faster compared to the
# others on one machine and slower on another machine. So when a program makes
# use of the same FFT algorithm many times, it pays off to invest a bit of time
# in the beginning to find an algorithm that runs fast. The function
# `WindowArrays.populate_fft_plan_cache` can be used to precompute a range of
# FFT plans. In the form below, the function takes three iterable arguments: The
# planning function names for the kinds of FFTs, the data types, and the input
# buffer sizes for which to plan. Plans will be constructed for all viable
# combinations. The planning functions can be found in the documentation of the
# `AbstractFFTs` and `FFTW` packages. E.g. `plan_brfft` plans a backwards real
# FFT and `plan_fft!` plans a forward complex in-place FFT. The buffer sizes
# used by `WindowArrays` are always integer powers of two.
using WindowArrays
populate_fft_plan_cache( # TODO
  [:plan_rfft, :plan_brfft],
  [Float64, ComplexF64],
  [2^12])

# Subsequent `mul_prepare` calls will now make use of the cached FFT plans
# instead of using inferior ad-hoc plans. When using this repository to run
# large data analyses or online training, it makes sense to run
# `populate_fft_plan_cache` once in the very beginning.


## Training the model

# So we have our feature matrix `a` and our label vector `b`, and we have a
# cache of ready-to-go FFT plans for fast matrix-vector multiplications. We can
# now train the model with the `fit` method. `fit` takes the model
# hyperparemeters, the feature matrix, and the label vector as arguments and
# returns the fitted model data in the form of a named tuple. We will set the
# keyword argument `allow_overwrite_a` to `false` so that `a` will not be
# modified. Its default value `true` may result in a small speedup, but does not
# guarantee that the data of `a` is preserved, as the name implies.
fw_fit = fit(fw, a, b; allow_overwrite_a = false)

# Notice that a couple of messages are printed during the evaluation of `fit`.
# The reason for this is that the `mul_prepare` call inside of `fit` decides to
# use FFTs of lengths `2^12` and `2^13`, but we have only precomputed FFTs of
# length `2^12`. This was of course by design to showcase this behavior. That
# `mul_prepare` call is configured to print a message whenever a required plan
# is not found in the cache. It is furthermore configured to not populate the
# cache on its own, because this could cause thread safety issues. The message
# serves as a reminder that we should add `2^13` to the list of FFT sizes in the
# above `populate_fft_plan_cache` call.


## Using/testing the trained model

# The fitted model data can now be used together with the hyperparameters to
# make predictions on a matrix of new features. As mentioned above, we will
# simply use the same matrix `a` as our "new features", even though this is bad
# practice.
b_predicted = apply(fw, a, fw_fit; allow_overwrite_a = false)

# Note that the prediction is dicrete and uses the labels from `fw.classes`. If
# we want to compare this prediction to something, we can not use `b`, because
# it contains labels for continuous lightness estimates. We can apply a
# thresholding to `b` using the threshold(s) in `fw.label_thresholds`:
b_thresholded = threshold_b(fw, b)

# This gives us a label vector that uses the same discrete labels as
# `b_predicted`. Now we can do a fair evaluation. Let's compute the
# classification accuracy:
fw_accuracy = count(b_predicted .== b_thresholded) / length(b)


### Using the segment-wise model

# We are going to use the model `TODO` to train a segment-wise classifier on our
# training dataset. We will then use the classifier to predict for each segment
# of our testing dataset whether it had an inverted phase or not. Here, the
# training and testing datasets will both be the same again (bad practice, just
# as above). They will consist of our frame-wise prediction `b_predicted` and
# the `segment_labels`.

# TODO

