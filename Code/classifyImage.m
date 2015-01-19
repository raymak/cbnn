function output = classifyImage(theta, visibleSize, hiddenSize, ...
                                             outputSize,lambda, sparsityParam, beta, data)

normalizedPatch = normalizeData(data);

output = hypothesisFunc(theta, visibleSize, hiddenSize, ...
                                             outputSize,lambda, sparsityParam, beta, normalizedPatch);



end
