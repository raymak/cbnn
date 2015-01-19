function [prediction] = predict(theta,data,opts)

    %get all parameters
    visibleSize = opts.visibleSize;
    outputSize = opts.outputSize;
    hiddenLayers = opts.hiddenLayers;
    hiddenSize = opts.hiddenSize;
    sparsityParam = opts.sparsityParam;
    lambda = opts.lambda;
    beta = opts.beta;
    neuronsPerBlock = opts.neuronsPerBlock;
    isBlocked = (neuronsPerBlock > 1); 


    %reshape weight parameters 
    index = 1;
    W = {};
    mat_size = hiddenSize(1) * visibleSize;
    W{1} = reshape(theta(index:index + mat_size - 1), hiddenSize(1), visibleSize);    
    index = index + mat_size;
    for i = 2:hiddenLayers     
        mat_size = hiddenSize(i) * hiddenSize(i-1); 
        W{i} = reshape(theta(index:index + mat_size - 1), hiddenSize(i), hiddenSize(i-1));    
        index = index + mat_size;
    end
    mat_size = hiddenSize(hiddenLayers) * outputSize;
    W{hiddenLayers + 1} = reshape(theta(index:index + mat_size - 1), outputSize, hiddenSize(hiddenLayers));    
    index = index + mat_size;
    %get bias vectors
    b = {};
    for i = 1:hiddenLayers
        b{i} = theta(index:index + hiddenSize(i)-1);
        index = index + hiddenSize(i);
    end
    b{hiddenLayers + 1} = theta(index:end);
   
    m = size(data,2); %number of samples
    z = {};
    a = {};
    z{1} = data; %X for each training example
    a{1} = data; %X for each training example
    for i = 2:hiddenLayers + 1
    z{i} = W{i-1} * a{i-1} + repmat(b{i-1},1,m);
    a{i} = activationFunc(z{i},neuronsPerBlock,isBlocked);
    end
    z{hiddenLayers + 2} = W{hiddenLayers + 1} * a{hiddenLayers + 1} + repmat(b{hiddenLayers + 1},1,m);
    a{hiddenLayers + 2} = outActivationFunc(z{hiddenLayers + 2});
    %a{hiddenLayers + 2} is the final hypothesis
    prediction = a{hiddenLayers+2};

end