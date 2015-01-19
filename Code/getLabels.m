function labelsVec = getLabels(is_test)
    %MNIST images
    addpath('..//MNIST');

    if is_test == 0
	    labels = loadMNISTLabels('..//MNIST//train-labels');
    else
	    labels = loadMNISTLabels('..//MNIST//test-labels');
    end

    %convert labels into 10 dimensional vectors
    labelsVec = zeros(10,length(labels));
    for i=1:length(labels)
        labelsVec(labels(i)+1,i) = 1;
    end

end
