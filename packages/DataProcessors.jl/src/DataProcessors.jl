"General abstractions for data processors making use of `fit`, `apply`, and/or
`apply!` methods. These are not necessarily machine learning models since there
is no requirement for `fit` to be implemented."
module DataProcessors

export
  DataProcessor,
  fit,
  apply,
  apply!,
  is_id

"Abstract supertype of any data processing specification types."
abstract type DataProcessor end

"Given data and a `DataProcessor`, fits the data processing model to the data
and returns the fit data."
function fit end

"Processes data according to the given `DataProcessor` and fit data."
function apply end

"Processes data like `apply`, but overwrites the original data."
function apply! end

"Checks whether the given `DataProcessor` is non-altering, that is, it turns
`apply` into an identity function on the processed data."
function is_id end

end
