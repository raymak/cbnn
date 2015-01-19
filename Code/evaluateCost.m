function [cost,grad] = evaluateCost(theta, visibleSize, hiddenLayers,hiddenSize, ...
                                             outputSize,lambda, sparsityParam, beta, neuronsPerBlock, batchSize,  data, labels)                                      

isBlocked = (neuronsPerBlock > 1);                                             

%Stochastic sample selection
if (batchSize > 0)
    selectedSamples = randi(size(data,2),batchSize,1);
    data = data(:,selectedSamples);
    labels = labels(:,selectedSamples);
end

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

%calculating the hypothesis - forward propagation
z = {};
a = {};
z{1} = data; %X for each training example
a{1} = data; %X for each training example
for i = 2:hiddenLayers + 1
z{i} = W{i-1} * a{i-1} + repmat(b{i-1},1,m);
a{i} = activationFunc(z{i},neuronsPerBlock,isBlocked);
end
z{hiddenLayers + 2} = W{hiddenLayers + 1} * a{hiddenLayers + 1} + repmat(b{hiddenLayers + 1},1,m);
a{hiddenLayers + 2} = activationFunc(z{hiddenLayers + 2},[], false);
%a{hiddenLayers + 2} is the final hypothesis

%calculating the cost
cost = 0.5 * sum(sum( (a{hiddenLayers + 2} - labels) .^ 2 )); %Loss function, %data - binarylabels
cost = cost / m; %normalizing

%calculatiion for sparsity term
%calculating rho_j for each hidden neuron j for each hidden layer
% rho = sparsityParam;
% rho_j = {};
% rho_j{1} = [];
% rho_j{hiddenLayers+2} = [];
% for i = 2:hiddenLayers+1
%     rho_j{i} = a{i};
%     %normalizing
%     rho_j{i} = sum(rho_j{i},2) ./ m;
% end

%backpropagation
%calculate delta's
delta = {};
for i = 1:hiddenLayers+2
    delta{i} = [];
end
delta{1} = [];
delta{hiddenLayers+2} = (a{hiddenLayers+2} - labels) .* activationFuncDerivative(z{hiddenLayers + 2},[],false); %data - labels
for i = hiddenLayers+1:-1:2
    delta{i} = (W{i}' * delta{i + 1}) .* activationFuncDerivative(z{i},neuronsPerBlock, isBlocked);
end
% %adding the term from sparsity constraint
% for i = 2:hiddenLayers+1
%     temp = repmat( beta * ( -rho * (1 ./ rho_j{i} ) + (1 - rho) * 1./ (1 - rho_j{i})  ), 1, m);  
%     delta{i} = delta{i} + temp  .* blockedWTAActivationDerivative(z{i},neuronsPerBlock);
% end

%gradient calculation
for i = hiddenLayers+1:-1:2
    Wgrad{i} = delta{i+1} * a{i}';
end
Wgrad{1} = delta{2} * data'; 
for i = hiddenLayers+1:-1:1
    bgrad{i} = bgrad{i} + sum(delta{i+1},2);
end

%adding contribution from regularization term
for i = 1:hiddenLayers+1
    cost = cost + (lambda/2) * sum(sum(W{i} .^ 2));
    Wgrad{i} = Wgrad{i} ./ m;
    Wgrad{i} = Wgrad{i} + lambda * W{i};
    bgrad{i} = bgrad{i} ./ m;
end

% %Adding the cost contribution from sparsity term
% for i = 2:hiddenLayers+1
%     cost = cost + beta * KL(rho,rho_j{i});
% end

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

%norm(grad)

end



function cost = KL(rho,rho_j)
	cost = sum(rho * log( rho * 1 ./ rho_j ) + (1 - rho) * log( (1 -rho) *  (1  ./  ( repmat(1,size(rho_j)) - rho_j)) ));
end

