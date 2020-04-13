%Author: Michael Busa
%ML HW 4 - Question 1
%4/7/20

%clear
clc

mult = 10;
%Generate datasets
C = 2; %Number of classes
n_train = 1000;
n_test = 10000;

[d_train, d_train_labels] = generateMultiringDataset(C,n_train);
[d_test, d_test_labels] = generateMultiringDataset(C,n_test);
%close all

%% 10-fold cross-validation
% Train a RBF kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
divs = 10; %number of possible C/sigma values to test
min_div = -3; %min value of C = 10^min_div
max_div = min_div+divs-1; %max value of C = 10^max_div

possible_c_vals = 10.^linspace(min_div,max_div,divs); %vector of possible C values
possible_sig_vals = 10.^linspace(min_div,max_div,divs); %vector of possible sigma values

folds = 10; %Number of folds
spf = n_train/folds; %Number of samples per fold
k_points =  n_train - spf; %Number of samples in the training set

%Split the data into 10 equal folds, and store labels accordingly
k_folds = zeros(2,spf,folds);
k_folds_labels = zeros(1,spf,folds);
for k=1:folds
    k_folds(:,:,k) = d_train(:,((k-1)*spf)+1:spf*k);
    k_folds_labels(:,:,k) = d_train_labels(:,((k-1)*spf)+1:spf*k);
end

p_correct = zeros(divs);
for c_counter = 1:divs %For all possible values of C
    C = possible_c_vals(c_counter);
    for sig_counter = 1:divs %For all possible values of sigma
        sig = possible_sig_vals(sig_counter);
        n_correct = zeros(1,folds);

        k_train = zeros(2,n_train-spf);
        k_train_labels = zeros(1,n_train-spf);
        for k = 1:folds
            fprintf('%d\t%d\t%d\n',c_counter,sig_counter,k);
            %Separate data into one train, one val
            unused_folds = [1:10];
            unused_folds(unused_folds==k) = [];
            k_val = k_folds(:,:,k); %Validation fold
            k_val_labels = k_folds_labels(:,:,k); %Labels of the validation fold
            for i=1:9
                k_train(:,(i-1)*spf+1:spf*i) = k_folds(:,:,unused_folds(i)); %Training Set
                k_train_labels(:,(i-1)*spf+1:spf*i) = k_folds_labels(:,:,unused_folds(i)); %Labels of the training set
            end

            % using all other folds as training set
            SVMk = fitcsvm(k_train',k_train_labels,'KernelFunction','rbf','BoxConstraint',C,'KernelScale',sig);
            val_label_predict = SVMk.predict(k_val')'; % Labels of validation data using the trained SVM
            n_correct(k)=length(find(k_val_labels==val_label_predict)); %Count the correctly predicted samples
        end 
    p_correct(c_counter,sig_counter)= sum(n_correct)/n_train; 
    end
end 
%%
for trials = 1:10
    [max_row,max_row_loc] = max(p_correct);
    [max_p_correct,best_sig_loc] = max(max_row);
    best_c_loc = max_row_loc(best_sig_loc);
    best_c = possible_c_vals(best_c_loc);
    best_sig = possible_sig_vals(best_sig_loc);

    %Train another SVM with the full training set and the best parameters
    clear SVMk
    SVMk = fitcsvm(d_train',d_train_labels,'KernelFunction','rbf','BoxConstraint',best_c,'KernelScale',best_sig);

    %Test on the Testing dataset
    test_label_SVM = SVMk.predict(d_test')'; % Labels of test data using the trained SVM
    correct=length(find(d_test_labels==test_label_SVM)); %Count the correctly predicted samples
    p_error = 100*(1-(correct/n_test));
    fprintf('Final Probability of Error on Test Set:\n\t%.4f\n',p_error);
end








