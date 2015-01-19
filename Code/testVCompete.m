%parameters
opts.trainSetSize = 60000;
opts.testSetSize = 10000;

opts.debug = 0;
opts.patch_size = 28;
opts.visibleSize = opts.patch_size ^ 2;   % number of input units 
opts.sparsityParam = 0.01;   % desired average activation of the hidden units.
opts.lambda = 3e-4;     % weight decay parameter       
opts.beta = 0;            % weight of sparsity penalty term       
opts.hiddenLayers = 2;  % number of hidden layers 
opts.hiddenSize = zeros(opts.hiddenLayers,1);
for i = 1:opts.hiddenLayers
	opts.hiddenSize(i) = 800;     % number of hidden units 
end

opts.hiddenSize = [1600; 800];
opts.neuronsPerBlock  = 2;
neuronsPerBlock = opts.neuronsPerBlock;
opts.batchSize = 0;    % 0 will consider all samples
opts.outputSize= 5; % number of output units
opts.maxiter = 1000;  %number of L-BFGS iterations
opts.tolfun = 1e-1;
opts.tolval = 4e-6;
opts.tolx = -1;
opts.maxfunevals = 10000;   %maximum function evaluations

%file save, load options
opts.savefile = 1; %whether to save the final set of weight parameters
opts.savefname = ['weights_P1' num2str(neuronsPerBlock)]; %save filename

opts.randomInitialize = 1; % whether to randomly initialize the weights or use previous weights
                           % as starting point for training if 0, specify filename of stored weights 
                           % in opts.loadfname
opts.randomInitializeOutput = 1;
                           
%load train data
[train_data, train_labels] = loadData(opts.patch_size,opts.trainSetSize,0);

%train the network on the entire network
% theta = train(train_data,train_labels,opts);

%train the network on only part of the data - training only for digits
%less than 5
[ma index] = max(train_labels,[],1);
train_labels_part = train_labels(1:5,index <= 5);
train_data_part = train_data(:,index <= 5);
%theta = train(train_data_part,train_labels_part,opts);

display('P1 Results');
load('weights_P11');
theta = opttheta;


%test the network on P1
[test_data, test_labels] = loadData(opts.patch_size,opts.testSetSize,1);
[ma index] = max(test_labels,[],1);
test_labels_part = test_labels(1:5,index <= 5);
test_data_part = test_data(:,index <= 5);

train_data_prediction = predict(theta,train_data_part,opts);
test_data_prediction = predict(theta,test_data_part,opts);

%calculate accuracy
[m index_trp] = max(train_data_prediction);
[m index_tep] = max(test_data_prediction);
[m index_trl] = max(train_labels_part);
[m index_tel] = max(test_labels_part);

P1_in_sample_accuracy =  (sum(index_trp == index_trl) / size(train_data_prediction,2)) * 100
P1_out_sample_accuracy = (sum(index_tep == index_tel) / size(test_data_prediction,2)) * 100
%%---------------------------------------------------------------------------------------------


opts.randomInitialize = 0;
opts.randomInitializeOutput = 1;
opts.loadfname = ['weights_P1' num2str(neuronsPerBlock)];
opts.savefname = ['weights_P2' num2str(neuronsPerBlock)];
opts.maxiter = 50;

count = 0;
totalCount = 25;

while count < totalCount
	
	count = count + 1;

	

	display('P2 Results');

	%test the network on P2
%	[test_data, test_labels] = loadData(opts.patch_size,opts.testSetSize,1);
	[ma index] = max(train_labels,[],1);
	train_labels_part = train_labels(6:10,index > 5);
	train_data_part = train_data(:,index > 5);  	
	[ma index] = max(test_labels,[],1);
	test_labels_part = test_labels(6:10,index > 5);
	test_data_part = test_data(:,index > 5);

	theta = train(train_data_part,train_labels_part,opts);

	train_data_prediction = predict(theta,train_data_part,opts);
	test_data_prediction = predict(theta,test_data_part,opts);

	%calculate accuracy
	[m index_trp] = max(train_data_prediction);
	[m index_tep] = max(test_data_prediction);
	[m index_trl] = max(train_labels_part);
	[m index_tel] = max(test_labels_part);

	P2_in_sample_accuracy =  (sum(index_trp == index_trl) / size(train_data_prediction,2)) * 100
	P2_out_sample_accuracy = (sum(index_tep == index_tel) / size(test_data_prediction,2)) * 100

%test the network for recall

	%combine weights
	load(['weights_P1' num2str(neuronsPerBlock)]);
	theta1 = opttheta;

	%reshape weight parameters 
	hiddenSize = opts.hiddenSize;
	visibleSize = opts.visibleSize;
	outputSize = opts.outputSize;
	hiddenLayers = opts.hiddenLayers;
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
	W{hiddenLayers + 1} = reshape(theta1(index:index + mat_size - 1), outputSize, hiddenSize(hiddenLayers));    
	index = index + mat_size;
	%get bias vectors
	b = {};
	for i = 1:hiddenLayers
	    b{i} = theta(index:index + hiddenSize(i)-1);
	    index = index + hiddenSize(i);
	end
	b{hiddenLayers + 1} = theta1(index:end);

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
	
	%load test data
%	[test_data, test_labels] = loadData(opts.patch_size,opts.testSetSize,1);
	
	%SAMPLE p1 SET
	
	[ma index] = max(train_labels,[],1);
	train_labels_part = train_labels(1:5,index <= 5);
	train_data_part = train_data(:,index <= 5);
	[ma index] = max(test_labels,[],1);
	test_labels_part = test_labels(1:5,index <= 5);
	test_data_part = test_data(:,index <= 5);

	%test the network on P1
	train_data_prediction = predict(theta,train_data_part,opts);
	test_data_prediction = predict(theta,test_data_part,opts);
	
	%calculate accuracy
	[m index_trp] = max(train_data_prediction);
	[m index_tep] = max(test_data_prediction);
	[m index_trl] = max(train_labels_part);
	[m index_tel] = max(test_labels_part);
	
	
	recall_in_sample_accuracy =  (sum(index_trp == index_trl) / size(train_data_prediction,2)) * 100
	recall_out_sample_accuracy = (sum(index_tep == index_tel) / size(test_data_prediction,2)) * 100


	opts.loadfname = ['weights_P2' num2str(neuronsPerBlock)];
	opts.randomInitializeOutput = 0;

end


%%---------------------------------------------------------------------------------------------

opts.loadfname = ['weights_P1' num2str(neuronsPerBlock)];


%combine weights
load(['weights_P1' num2str(neuronsPerBlock)]);
theta1 = opttheta;

%reshape weight parameters 
hiddenSize = opts.hiddenSize;
visibleSize = opts.visibleSize;
outputSize = opts.outputSize;
hiddenLayers = opts.hiddenLayers;
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
W{hiddenLayers + 1} = reshape(theta1(index:index + mat_size - 1), outputSize, hiddenSize(hiddenLayers));    
index = index + mat_size;
%get bias vectors
b = {};
for i = 1:hiddenLayers
    b{i} = theta(index:index + hiddenSize(i)-1);
    index = index + hiddenSize(i);
end
b{hiddenLayers + 1} = theta1(index:end);

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

%load test data
[test_data, test_labels] = loadData(opts.patch_size,opts.testSetSize,1);

%SAMPLE p1 SET
[ma index] = max(train_labels,[],1);
train_labels_part = train_labels(1:5,index <= 5);
train_data_part = train_data(:,index <= 5);
[ma index] = max(test_labels,[],1);
test_labels_part = test_labels(1:5,index <= 5);
test_data_part = test_data(:,index <= 5);

%test the network on P1
train_data_prediction = predict(theta,train_data_part,opts);
test_data_prediction = predict(theta,test_data_part,opts);

%calculate accuracy
[m index_trp] = max(train_data_prediction);
[m index_tep] = max(test_data_prediction);
[m index_trl] = max(train_labels_part);
[m index_tel] = max(test_labels_part);


recall_in_sample_accuracy =  (sum(index_trp == index_trl) / size(train_data_prediction,2)) * 100
recall_out_sample_accuracy = (sum(index_tep == index_tel) / size(test_data_prediction,2)) * 100

display(['Reporting all accuracies'])
P1_in_sample_accuracy
P1_out_sample_accuracy
P2_in_sample_accuracy
P2_out_sample_accuracy
recall_in_sample_accuracy
recall_out_sample_accuracy
