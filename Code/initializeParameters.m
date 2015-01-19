function theta = initializeParameters(hiddenSize, visibleSize,outputSize)

    hiddenLayers = size(hiddenSize,1);
    %% Initialize parameters randomly based on layer sizes.
    
    W = {}; %use cell array to store all the weight matrices and bias vectors
    
    r = sqrt(6) / sqrt(hiddenSize(1) + visibleSize + 1); % we'll choose weights uniformly from the interval [-r, r]
    W{1} = rand(hiddenSize(1), visibleSize) * 2 * r - r;
    b{1} = zeros(hiddenSize(1), 1);

    for i = 2:hiddenLayers %number of weight matrices
        r = sqrt(6) / sqrt(hiddenSize(i) + hiddenSize(i-1) + 1);
        W{i} = rand(hiddenSize(i-1), hiddenSize(i)) * 2 * r - r;
        b{i} = zeros(hiddenSize(i), 1);
    end
    
    r = sqrt(6) / sqrt(hiddenSize(hiddenLayers) + outputSize + 1);
    W{hiddenLayers + 1} = rand(hiddenSize(hiddenLayers), outputSize) * 2 * r - r;  
    b{hiddenLayers + 1} = zeros(outputSize, 1);

    % Convert weights and bias gradients to the vector form.
    % This step will "unroll" (flatten and concatenate together) all 
    % your parameters into a vector, which can then be used with minFunc. 
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

