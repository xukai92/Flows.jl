using CuArrays: @cufunc

@cufunc logit(x) = log(x) - log(1 - x)