# This is a tutorial meant to be run line by line e.g. in a Julia REPL or
# Jupyter notebook, so that the result of every step can be inspected and
# understood. It introduces the basic usage of the frame-wise and segment-wise
# machine learning models.

# If you have Julia running, make sure the directory this `tutorial.jl` file is
# contained in has been activated as the current project. You can do this with
# `] activate path/to/project`.

# Also make sure that the project has been instantiated. This can be done with
# `] instantiate`.

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
  [-.2, .6, -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -.6, .2]
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
# stimulus with its 3072 frames would then have a duration of 30.72s. Yes, this
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
# There may also be something to gain from other kinds of preprocessing, such as
# amplitude normalization of channels or spatial filtering.

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
window_size = samples_per_segment / samples_per_second
signal_offset = (vep_delay - .5 * samples_per_segment) / samples_per_second
fw = FwSLT(
  window_size,
  signal_offset,
  1.,
  :μ_σ,
  false,
  Binary,
  (false, true),
  (.0,);
  atol = 1e-6,
  btol = 1e-6)

# The `FwSLT` model implements the frame-wise sliding window classifier as
# described in my Master's Thesis (and very similar to what Martin Spüler,
# Sebastian Nagel, Alexander Blöck, and probably others have been doing prior to
# that). It classifies the lightness value of individual time points by placing
# a window around them in the EEG signal and running three stages of machine
# learning on them.

# In the above, as a starting point, the windows are set to have the same size
# as one CCVEP segment, and to be centered around the time point in question,
# respecting the `vep_delay` (that is, the (usually unknown) time it takes for a
# change on the screen to show up as a VEP in the EEG).

# The first machine learning stage that the model performs on each window is a
# feature standardization by affine (linear) transformation. The `:μ_σ` above
# specifies that the features should be standardized to have a mean of zero and
# a variance of 1.

# The second stage is a ridge regression. With the above parameters, the ridge
# regression will have a λ parameter of 1 and no bias term. I have usually not
# seen performance improvements from incorporating a bias term in the feature
# matrix, though I suspect that this would change when the EEG siganl is not
# highpass-filtered or when the label vector has a mean that is significantly
# different from zero.

# The third stage performs a thresholding to convert continuous lightness values
# to discrete labels such as "black" and "white" (represented by `false` and
# `true` in this case).

# The `atol` and `btol` are stopping criteria for the optimization algorithm
# used for the ridge regression . Using an EEG signal / feature matrix in 64-bit
# floating point precision and the tolerance value of `1e-6` has usually
# resulted in a good trade-off between accuracy of the solution and runtime for
# me. More on this further below.

# To construct a feature matrix (`fw_a`) and a label vector (`fw_b`) from our
# input data, we will use the `ab` method for the `FwSLT` type. This method
# takes, among its other arguments, an object `label_markers` that indicates the
# time points in the signal where windows should be placed, and what label to
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
fw_a, fw_b = ab(fw, eeg, label_markers, samples_per_second)

# When evaluating `fw_a`, we can see in the console output that its type is a
# `WindowArrays.WindowMatrix`. This is a custom matrix type that internally only
# stores a vector of data (`fw_a.v`), a vector of indexes into the data vector
# (`fw_a.i`), and the number of matrix columns (`fw_a.k`). The matrix has one
# row for every element in `fw_a.i`, which is a view of length `fw_a.k` into
# `fw_a.v`.

# The `ab` method flattens the `eeg` signal matrix into a vector such that each
# channel is enumerated for the first time point, then each channel for the
# second time point, and so on. This is why the number of columns of `fw_a` is
# the window size in samples multiplied by the number of channels. Each row is a
# window into the flattened signal matrix where the first `m` values represent
# the first time point included in the window, the following `m` values
# represent the next time point, etc. To access only the channel `j` from a row
# `i`, one would use something like `fw_a[i, j : m : end]` or
# `view(fw_a, i, j : m : lastindex(fw_a, 2))`.

# In the case of overlapping windows, the `WindowMatrix` is more space-efficient
# than a standard matrix that stores each of its elements individually (and
# hence redundantly). It also allows for a faster matrix multiplication that
# makes use of the fast Fourier transform (FFT) algorithm. Some steps of the
# fast matrix multiplication can be precomputed to be reused in all subsequent
# multiplications. This is done with the `WindowArrays.mul_prepare` function.
# The `WindowMatrix` object has two more fields in addition to those mentioned
# above: `fw_a.mp` and `fw_a.mpa`, which hold the precomputed data for
# multiplications of the form `fw_a * x` and `fw_a' * x`, respectively. Before
# calling `mul_prepare`, those `fw_a.mp` and `fw_a.mpa` will both be `nothing`.
# We will not call `mul_prepare` manualy here, because it will be done inside
# the `fit` method that we will invoke further below, but the call would look
# something like this:
#using WindowArrays: mul_prepare
#fw_a = mul_prepare(fw_a)

# Note that at the time of writing this, the `mpa` field won't be set by
# `mul_prepare`, because the FFT-based multiplication code for the `fw_a' * x`
# case produces erroneous results, currently. It's no big deal though, because
# the non-FFT-based version is also efficient and precise. The `mp` field is the
# important one.

# As mentioned above, I usually used 64-bit floating point feature matrices. The
# reason for this was that in the least-squares procedure, when using 32-bit,
# the cumulative numeric error got too large for the algorithm to converge to a
# good solution. After migrating the code to this repository, I have improved
# the numerics of the multiplication functions for `WindowMatrix`. For the kind
# of matrix sizes we're using here, the numerics are now even better than those
# of the BLAS routines used for the standard `Matrix` types. So perhaps, by now,
# a `WindowMatrix{Float32}` is sufficient to achieve good solutions. It may be
# worth experimenting with that – I didn't have the time to test this myself.

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
populate_fft_plan_cache(
  [:plan_rfft, :plan_brfft],
  [Float64, ComplexF64],
  [2^12])

# Subsequent `mul_prepare` calls will now make use of the cached FFT plans
# instead of using inferior ad-hoc plans. When using this repository to run
# large data analyses or online training, it makes sense to run
# `populate_fft_plan_cache` once in the very beginning.

# Lastly, in case there are any problems with the `WindowMatrix` type: The model
# works just fine with the standard `Matrix` type as well. A `WindowMatrix` can
# be converted to a `Matrix` like this:
#fw_a = materialize(fw_a)
# Note, however, that the "materialized" `Matrix` may have quite the memory
# footprint.


## Training the model

# So we have our feature matrix `fw_a` and our label vector `fw_b`, and we have
# a cache of ready-to-go FFT plans for fast matrix-vector multiplications. We
# can now train the model with the `fit` method. `fit` takes the model
# hyperparemeters, the feature matrix, and the label vector as arguments and
# returns the fitted model data in the form of a named tuple. We will set the
# keyword argument `allow_overwrite_a` to `false` so that `fw_a` will not be
# modified. Its default value `true` may result in a small speedup, but does not
# guarantee that the data of `fw_a` is preserved, as the name implies.
fw_fit = fit(fw, fw_a, fw_b; allow_overwrite_a = false)

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
# simply use the same matrix `fw_a` as our "new features", even though this is
# bad practice.
fw_b_predicted = apply(fw, fw_a, fw_fit; allow_overwrite_a = false)

# Note that the prediction is dicrete and uses the labels from `fw.classes`. If
# we want to compare this prediction to something, we can not use `fw_b`,
# because it contains labels for continuous lightness estimates. We can apply a
# thresholding to `fw_b` using the threshold(s) in `fw.label_thresholds`:
fw_b_thresholded = threshold_b(fw, fw_b)

# This gives us a label vector that uses the same discrete labels as
# `fw_b_predicted`. Now we can do a fair evaluation. Let's compute the
# classification accuracy:
fw_accuracy = count(fw_b_predicted .== fw_b_thresholded) / length(fw_b)


### Using the segment-wise model

# We are going to use the model `SwSLT` to train a segment-wise classifier on
# our training dataset. We will then use the classifier to predict for each
# segment of our testing dataset whether it had an inverted phase or not. Here,
# the training and testing datasets will both be the same again (bad practice,
# just as above). They will consist of our frame-wise prediction
# `fw_b_predicted` and the `segment_labels`.


## Preparing the model inputs

using VepModelSwSLT

# Because the `SwSLT` model is so similar to the `FwSLT` model, it uses almost
# the same set of hyperparameters. Instead of the window size and signal offset,
# it takes the segment length as its first argument. Also, `label_thresholds` is
# optional because it can be deduced from the `classes`, if they are numerical
# (which they are in this case, because just like in Python, `Bool`s are
# considered numbers equal to 0 or 1 in Julia). Something non-numerical like
# strings could be used as classes too for these models, but it wouldn't really
# add any value in this case. We will use a bias term in this case, because the
# mean of our `segment_labels` is .5, i.e. nonzero. We could instead also have
# used `-1` and `1` in the `segment_labels` definition.
sw = SwSLT(
  frames_per_segment,
  1.,
  :μ_σ,
  true,
  Binary,
  (false, true);
  atol = 1e-6,
  btol = 1e-6)

# The `ab` method for the `SwSLT` model takes a vector of lightness values and a
# vector of segment labels as inputs. We already have our `segment_labels`, and
# we will use our previous prediction `fw_b_predicted` as lightness values.

# Of course, the latter have been thresholded to `false` and `true`.
# Technically, the model also allows for the use of continuous lightness values,
# but using the thresholded values tends to slightly improve the classification
# accuracies.

# In our case here, in `fw_b_predicted`, all frames are ordered chronologically,
# because we never permuted the ordering. If we had done a cross-validation with
# shuffling, we would need to reconstruct the correct ordering first (or order
# `segment_labels` accordingly). The `VepModels.cv_splits` function can help
# with this. This is fairly common functionality, however, that can also be
# found in machine learning packages.

# So let's create the feature matrix and label vector:
sw_a, sw_b = ab(sw, fw_b_predicted, segment_labels)


## Training the model

# This should look familiar:
sw_fit = fit(sw, sw_a, sw_b; allow_overwrite_a = false)


## Using/testing the trained model

# And this is also done the same way as for the frame-wise model:
sw_b_predicted = apply(sw, sw_a, sw_fit; allow_overwrite_a = false)

# We could compute a `sw_b_thresholded` analogously to above, but in this case,
# it would just be exactly the same as `sw_b`. So we can just use that one in
# the accuracy calculation:
sw_accuracy = count(sw_b_predicted .== sw_b) / length(sw_b)


### Maximum length sequence (m-sequence) shift classification

# In my thesis, the bit string encoded by the segments in the stimuli was a
# maximum length sequence (MLS). MLSs have been used in cVEP-based BCIs, because
# their autocorrelation property can be used to present the same MLS with
# multiple distinct time shifts (rotation offsets) and then classify the shift.
# The same can be done with the segment-wise predictions here, if the segments'
# bits are an MLS. Furthermore, Expanding the segments into the full vector of
# lightness values mostly preserves the autocorrelation property, and hence, the
# MLS shift classification can also be done on the frame-wise predictions
# directly, which seems to lead to better classification accuracies.

# At the time of writing this, this repository contains no code specifically for
# MLS shift classification. However, this is fairly easy to do by hand. To
# illustrate this here briefly, let's say our original 192-element bit string
# `segment_labels` is really a concatenation of 64 stimuli of length 3, namely
# 010, 101, 010, etc. These are MLSs of length 3. Let's define the following:

mls0 = BitVector((false, false, true))
mls1 = (!).(mls0)

# So our first MLS with shift 0 (`mls0`) is now defined to be 001, and our
# second MLS with shift 0 (`mls1`) is the inversion of the first, 110. A MLS
# shift classifier as it is usually implemented can only distinguish between
# different shifts. Mixing an MLS with its inverted version will confuse such a
# classifier. So basically, we pretend here that we did an experiment where we
# tested two distinct MLSs (`mls0` and `mls1`), and we presented 32 trials of
# each, alternating between the two MLSs, and we always presented them with a
# shift (to the right) of 2. I hope this makes sense. Ideally, the experiment
# should also have presented the MLSs with shifts 0 and 1, in randomized order,
# probably.

# Let's extract the dataset of 32 trials for `mls0` into a matrix where each row
# represents one trial, i.e. one MLS:
sw_mlss_predicted = reshape(sw_b_predicted, (3, :))'
sw_mls0s_predicted =
  view(sw_mlss_predicted, 1 : 2 : lastindex(sw_mlss_predicted, 1), :)

# We now want to cross-correlate each row with `mls0` and find the maximum. In
# other words, we will correlate each row with `mls0` shifted by 0, by 1, and by
# 2, and see which shift yields the largest value. There are different
# definitions of correlation, e.g. the Pearson correlation coefficient, but a
# simple dot product should suffice in this case. We can do all three dot
# products for a row at once using a matrix:
mls0_shifts = hcat((circshift(mls0, i) for i in 0:2)...)

# The first column of `mls0_shifts` is `mls0` with shift 0, the second column is
# `mls0` with shift 1, and the third with shift 2. Multiplying
# `sw_mls0s_predicted` with `mls0_shifts` now gives us the cross-correlations of
# each trial with each shifted `mls0`. The `argmax` of each row gives us the
# index where the cross-correlation is largest, and subtracting 1 gives us the
# shift (because of Julia's 1-based indexing – ahh, everything would be so much
# cleaner if they had chosen 0-based).
sw_mls0s_shifts_predicted = [argmax(row) - 1
  for row in eachrow(sw_mls0s_predicted * mls0_shifts)]

# What would be our ground truth, i.e. labels, here? It would be a vector of the
# shifts with which `mls0` was presented in the trials. So here in this
# contrived example, it would be a 32-element vector of twos.
sw_mls0s_shifts = fill(2, size(sw_mls0s_predicted, 1))

# Now we can calculate the classification accuracy:
sw_mls0_accuracy =
  count(sw_mls0s_shifts_predicted .== sw_mls0s_shifts) / length(sw_mls0s_shifts)

# For the frame-wise MLS classifier, the procedure would be pretty much the
# same, except that vectors/matrices of lightness values would be used instead
# of segment bit values, and shifts would be done in integer multiples of the
# segment length.


### Multi-threading

# In the use case of large cross-validated hyperparameter searches, it may make
# sense to assign a single thread per model being trained. Conversely, in the
# use case of training a single model in an online setting, any parallelization
# would have to happen on a lower level. The LSMR least-squares algorithm used
# by the models is single-threaded. I think it should be relatively simple to
# substitute the `lsmr!` call in the code by a call to some multi-threaded
# algorithm, if desired. (The algorithm would have to accept an `AbstractMatrix`
# and not use it for much else than multiplying it or its transpose with
# vectors.) However, parallelization at the level of the matrix multiplications
# is also an option.

# The second line of the below code gets the number of threads of the current
# Julia process. If the `julia` command was run with the `--threads=auto`
# argument, it should be equal to the total number of CPU threads. In the last
# two lines, the number of threads is set for BLAS and `WindowArrays`. BLAS is
# the library that performs the linear algebra operations for the standard
# `Matrix` types. Setting these to 1 disables multi-threaded matrix
# multiplications.
import LinearAlgebra, Base.Threads, WindowArrays
n_threads = Threads.nthreads()
LinearAlgebra.BLAS.set_num_threads(n_threads)
WindowArrays.num_threads(n_threads)


# I hope this tutorial will be helpful.
(; fw_accuracy, sw_accuracy, sw_mls0_accuracy)

