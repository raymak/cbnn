function sigm_der = sigmoidDerivative(x)
    sigm_der = sigmoid(x) .* (repmat(1,size(x)) - sigmoid(x));

end