%Author: Michael Busa
%ML HW 4 - Question 1
%4/7/20

clear
clc

%Generate datasets
C = 4;
n_train = 1000;
n_test = 10000;

d_train = exam4q1_generateData(n_train);
d_test = exam4q1_generateData(n_test);
%close all
%Activation Functions
%Softplus: ln(1+e^x)

%% 10-fold cross-validation
for tries = 1:10
    clearvars -except d_test d_train n_train n_test C
    nX = 1; %Number of inputs
    nY = 1; %Number of outputs
    P = 6; %How many perceptrons to test up to, 1-P

    folds = 10; %Number of equal part folds
    spf = n_train/folds; %Number of samples in each fold
    k_points =  n_train - spf; %Number of samples in the training set

    %Split the data set into 10 equal parts
    k_folds = zeros(2,spf,folds);
    for k=1:folds
        k_folds(:,:,k) = d_train(:,((k-1)*spf)+1:spf*k);
    end

    k_train = zeros(2,n_train-spf);
    results = zeros(1,P);
    best_params = zeros(19,P);
    for p = 1:P %perceptrons
        total_folds_mse = 0;
        for k=1:folds
            %Separate data into one train, one val
            unused_folds = [1:10];
            unused_folds(unused_folds==k) = [];
            k_val = k_folds(:,:,k);
            for i=1:9
                k_train(:,(i-1)*spf+1:spf*i) = k_folds(:,:,unused_folds(i));
            end
            % Determine/specify sizes of parameter matrices/vectors
            nPerceptrons = p; 
            sizeParams = [nX;nPerceptrons;nY];       

            % Initialize model parameters
            params.A = 10*rand(nPerceptrons,nX);
            params.b = 10*rand(nPerceptrons,1);
            params.C = 10*rand(nY,nPerceptrons);
            params.d = 10*randn(nY,1);
            vecParamsInit = [params.A(:);params.b;params.C(:);params.d];

            % Optimize model
            x_in = k_train(1,:); %Input for Neural Net
            x_out = k_train(2,:); %True/desired output for Neural Net
            options = optimset('MaxFunEvals',200000, 'MaxIter', 200000);
            vecParams = fminsearch(@(vecParams)(objectiveFunction(x_in,x_out,sizeParams,vecParams)),vecParamsInit,options);

            clear params
            params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
            params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
            params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
            params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);

            %Test on validator set
            x_in_val = k_val(1,:); %Input of validation set
            x_out_val_true = k_val(2,:); %true/desired output of validation set
            x_out_mlp = mlpModel(x_in_val,params); %mlp output
            val_mse = sum((x_out_val_true-x_out_mlp).^2)/spf; %MSE
            total_folds_mse = total_folds_mse + val_mse; %Cumulative sum of MSE for the full cross validation

            %Save parameters for later
            if k==1
                min_mse = val_mse;
                best_params(1:length(vecParams),p) = vecParams;
            elseif val_mse<min_mse
                min_mse = val_mse;
                best_params(1:length(vecParams),p) = vecParams;
            end
        end
        results(p) = total_folds_mse/10;
        fprintf('Perceptrons: %d\tAverage MSE: %.4f\n',p,results(p));
    end

    best_p = find(results == min(results));
    % Train model for best p on entire training set
    clear x_in x_out sizeParams vecParams
    sizeParams = [nX;best_p;nY];
    x_in = d_train(1,:); %Entire training set as input
    x_out = d_train(2,:); %True training set output
    param_length = 3*best_p+1;
    vecParamsInit_final = best_params(1:param_length,best_p); %Using the parameters found during k-fold as initial values
    vecParams = fminsearch(@(vecParams)(objectiveFunction(x_in,x_out,sizeParams,vecParams)),vecParamsInit_final,options);

    clear params
    params.A = reshape(vecParams(1:nX*best_p),best_p,nX);
    params.b = vecParams(nX*best_p+1:(nX+1)*best_p);
    params.C = reshape(vecParams((nX+1)*best_p+1:(nX+1+nY)*best_p),nY,best_p);
    params.d = vecParams((nX+1+nY)*best_p+1:(nX+1+nY)*best_p+nY);

    %Apply MLP to Dtest;
    x_in_test = d_test(1,:);
    x_out_test_true = d_test(2,:);
    x_test_mlp = mlpModel(x_in_test,params);
    final_mse = sum((x_out_test_true-x_test_mlp).^2)/n_test;
    fprintf('\nBest Number of Perceptrons: %d\nFinal MSE: %2.4f\n',best_p,final_mse);
end


