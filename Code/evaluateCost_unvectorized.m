function [cost,grad] = evaluateCost(theta, visibleSize, hiddenLayers,hiddenSize, ...
    outputSize,lambda, sparsityParam, beta, data, labels)

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


%Initialization
cost = 0;
Wgrad = {};
bgrad = {};
for i = 1:hiddenLayers+1
    Wgrad{i} = zeros(size(W{i}));
    bgrad{i} = zeros(size(b{i}));
end

m = size(data,2); %number of samples

for count = 1:m
    
    %calculating the hypothesis - forward propagation
    z = {};
    a = {};
    z{1} = data(:,count); %X for each training example
    a{1} = data(:,count); %X for each training example
    for i = 2:hiddenLayers + 2
        z{i} = W{i-1} * a{i-1} + b{i-1};
        a{i} = activationFunc(z{i});
    end
    %a{hiddenLayers + 2} is the final hypothesis
  
    %calculating the cost
    cost = 0.5 * sum(sum( (a{hiddenLayers + 2} - labels(:,count)) .^ 2 )); %Loss function, %data -labels
         
    %backpropagation
    %calculate delta's
    delta = {};
    for i = 1:hiddenLayers+2
        delta{i} = [];
    end
    
    delta{1} = [];
    delta{hiddenLayers+2} = (a{hiddenLayers+2} - labels(:,count)) .* sigmoidDerivative(z{hiddenLayers + 2}); %data - labels
    for i = hiddenLayers+1:-1:2
        delta{i} = (W{i}' * delta{i + 1}) .* sigmoidDerivative(z{i});
    end
    
    %gradient calculation
    for i = hiddenLayers+1:-1:2
        Wgrad{i} = Wgrad{i} + delta{i+1} * a{i}';
    end
    
    Wgrad{1} = delta{2} * data(:,count)';
    for i = hiddenLayers+1:-1:1
        bgrad{i} = bgrad{i} + sum(delta{i+1},2);
    end
    
end

%adding contribution from regularization term
for i = 1:hiddenLayers+1
    cost = cost / m; %normalizing
    cost = cost + (lambda/2) * sum(sum(W{i} .^ 2));
    Wgrad{i} = Wgrad{i} ./ m;
    Wgrad{i} = Wgrad{i} + lambda * W{i};
    bgrad{i} = bgrad{i} ./ m;
end

%vectorization
grad = [];
%vectorizing weight matrices
for i = 1:hiddenLayers + 1
    grad = [grad ; Wgrad{i}(:)];
end
%vectorizing bias vectors
for i = 1:hiddenLayers + 1
    grad = [grad ; bgrad{i}(:)];
end



end

