function [opttheta] = train(data,labels,opts)

size(data)
size(labels)
%get all parameters
visibleSize = opts.visibleSize;
outputSize = opts.outputSize;
hiddenLayers = opts.hiddenLayers;
hiddenSize = opts.hiddenSize;
sparsityParam = opts.sparsityParam;
lambda = opts.lambda;
beta = opts.beta;
neuronsPerBlock = opts.neuronsPerBlock;
batchSize = opts.batchSize;
maxiter = opts.maxiter;
tolfun = opts.tolfun;
tolx = opts.tolx;
tolval = opts.tolval;
maxfunevals = opts.maxfunevals;
debug = opts.debug;

if debug == 1
    
    %  Obtain random parameters theta
    theta = initializeParameters(hiddenSize, visibleSize, outputSize);
    
    %function to find the cost and the gradient at a particular theta
    [cost, grad] = evaluateCost(theta, visibleSize, hiddenLayers,hiddenSize,outputSize,lambda, ...
        sparsityParam, beta, neuronsPerBlock, batchSize, data, labels);
    
    checkNumericalGradient();
    % Now we can use it to check your cost function and derivative calculations
    % for the sparse autoencoder.
    numgrad = computeNumericalGradient( @(x) evaluateCost(x, visibleSize, ...
        hiddenLayers,hiddenSize,outputSize,lambda, ...
        sparsityParam, beta,neuronsPerBlock, ...
        batchSize, data, labels), theta);
    
    % Use this to visually compare the gradients side by side
    disp([numgrad grad]);
    % Compare numerically computed gradients with the ones obtained from backpropagation
    diff = norm(numgrad-grad)/norm(numgrad+grad);
    fprintf('Diff is ')
    disp(diff); % Should be small. In our implementation, these values are
    % usually less than 1e-9.
    
end

%%======================================================================

%  Randomly initialize the parameters
if opts.randomInitialize == 1
    theta = initializeParameters(hiddenSize, visibleSize,outputSize);
else
    theta1 = initializeParameters(hiddenSize, visibleSize,outputSize);
    load(opts.loadfname,'opttheta');
    theta = opttheta;
    
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

    if opts.randomInitializeOutput == 1 
	W{hiddenLayers + 1} = reshape(theta1(index:index + mat_size - 1), outputSize, hiddenSize(hiddenLayers));
    else
	W{hiddenLayers + 1} = reshape(theta(index:index + mat_size - 1), outputSize, hiddenSize(hiddenLayers));
    end

    index = index + mat_size;
    %get bias vectors
    b = {};
    for i = 1:hiddenLayers
        b{i} = theta(index:index + hiddenSize(i)-1);
        index = index + hiddenSize(i);
    end
    if opts.randomInitializeOutput == 1
	b{hiddenLayers + 1} = theta1(index:end);
    else 
	b{hiddenLayers + 1} = theta(index:end);
    end
        

    %mixed theta
    theta = [];
    %vectorizing weight matrices
    for i = 1:hiddenLayers + 1
        theta = [theta ; W{i}(:)];
    end
    %vectorizing bias vectors
    for i = 1:hiddenLayers + 1
        theta = [theta ; b{i}(:)];
    end
    
end

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'bb'; % Here, we use L-BFGS to optimize our cost
% function. Generally, for minFunc to work, you
% need a function pointer with two outputs: the
% function value and the gradient. In our problem,
% sparseAutoencoderCost.m satisfies this.
options.maxIter = maxiter;	  % Maximum number of iterations of L-BFGS to run
options.display = 'on';
options.tolFun = tolfun;
options.tolX = tolx;
options.tolVal = tolval;
options.maxFunEvals = maxfunevals;

[opttheta, cost] = minFunc( @(p) evaluateCost(p, ...
    visibleSize, hiddenLayers,hiddenSize,outputSize, ...
    lambda, sparsityParam, ...
    beta, neuronsPerBlock, batchSize, data, labels), ...
    theta, options);
if opts.savefile == 1
    save(opts.savefname,'opttheta')
end

end
