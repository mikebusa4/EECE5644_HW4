%Author: Michael Busa
%ML HW 4 - Question 3
%4/7/20


clc
clear

regWeight = 1e-10;
delta = 1e-2;
Runs = 10;

%Read in an image
%filename = '3096_color.jpg';  %Plane
filename = '42049_color.jpg'; %Bird
pic = imread(filename);
figure(21)
imshow(pic);
title('Bird')

%Calculate pixel size of image
[row,col,~] = size(pic);
row_data = zeros(1,row*col);
col_data = zeros(1,row*col);
index = 1;
for i=1:row
    for j=1:col
        row_data(index) = i;
        col_data(index) = j;
        index = index+1;
    end
end

rgb = impixel(pic,col_data,row_data); %get rgb data for every pixel
red = rgb(:,1); %get red data
green = rgb(:,2); %get green data
blue = rgb(:,3); %get blue data
pixels = row*col; %calculate total number of pixels

f_vec = zeros(pixels,5);
f_vec(:,1) = row_data(:); %Column 1 holds pixel row
f_vec(:,2) = col_data(:); %Column 2 holds pixel column
f_vec(:,3) = red(:); %Column 3 holds red value
f_vec(:,4) = green(:); %Column 4 holds green value
f_vec(:,5) = blue(:); %Column 5 holds blue value

%Normalize each column individually on interval [0,1]
for i=1:pixels   
    %f_vec(i,:) = normalize(f_vec(i,:),'range');
end

data = f_vec';
data = data(3:5,:);

%% GMM Fitting
N = pixels;
% Initialize the GMM to randomly selected samples
gmm_c = 2;
d = 3;
for x = 1:Runs
    alpha = ones(1,gmm_c)/gmm_c;
    shuffledIndices = randperm(N);
    mu = data(:,shuffledIndices(1:gmm_c)); % pick M random samples as initial mean estimates
    [~,assignedCentroidLabels] = min(pdist2(mu',data'),[],1); % assign each sample to the nearest mean
    for m = 1:gmm_c % use sample covariances of initial assignments as initial covariance estimates
        Sigma(:,:,m) = cov(data(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
        if(isnan(Sigma(:,:,m)))
            Sigma(:,:,m) = eye(d,d);
        end
    end

    t = 0;
    Converged = 0; % Not converged at the beginning
    while ~Converged
        for l = 1:gmm_c
            temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(data,mu(:,l),Sigma(:,:,l));
        end
        plgivenx = temp./sum(temp,1);
        alphaNew = mean(plgivenx,2);
        w = plgivenx./repmat(sum(plgivenx,2),1,N);
        muNew = data*w';
        for l = 1:gmm_c
            v = data-repmat(muNew(:,l),1,N);
            u = repmat(w(l,:),d,1).*v;
            SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
        end
        Dalpha = sum(abs(alphaNew-alpha));
        Dmu = sum(sum(abs(muNew-mu)));
        DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
        Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
        alpha = alphaNew; 
        mu = muNew; 
        Sigma = SigmaNew;
        t = t+1;
    end
    alpha_total = alpha';
    mu_total = mu;
    Sigma_total = Sigma;

    results = zeros(2, pixels);
    for i=1:2
        %likelihood values for every sample for each class
        results(i,:) = evalGMM(data,alpha_total(i),mu_total(:,i), Sigma_total(:,:,i));
    end
    %Which ever row had a higher value is the class label
    [~,labels] = max(results);

    c1_2 = find(labels==1); %Indices in feature vector of class 1 data
    c2_2 = find(labels==2); %Indices in feature vector of class 2 data

    figure(x)
    m=3;
    plot(col_data(c1_2),-row_data(c1_2),'k.','Markersize',m)
    hold on
    plot(col_data(c2_2),-row_data(c2_2),'c.','Markersize',m)
end

%% Multiple Components with 10-fold cross validatiaon
N_train = pixels-1;
d = 3;
folds = 10;
spf = N_train/folds;
k_points =  N_train - spf;

k_folds = zeros(d,spf,folds);
for k=1:folds
    k_folds(:,:,k) = data(:,((k-1)*spf)+1:spf*k);
end

GMM_C = 3;
k_train = zeros(d,N_train-spf);

hoods = zeros(GMM_C,folds);
for gmm_c = 1:GMM_C %GMM components
    for k=1:folds
        %Separate data into one train, one val
        fprintf('%d\t%d\n',gmm_c,k);
        unused_folds = [1:10];
        unused_folds(unused_folds==k) = [];
        k_val = k_folds(:,:,k);
        for i=1:9
            k_train(:,(i-1)*spf+1:spf*i) = k_folds(:,:,i);
        end

        N = length(k_train);
        clearvars -except gmm_c Runs row_data col_data GMM_C Runs k alpha_total mu_total Sigma_total folds k_folds d hoods comp_choice N pixels k_train data regWeight delta k_val spf
        % Initialize the GMM to randomly selected samples
        alpha = ones(1,gmm_c)/gmm_c;
        shuffledIndices = randperm(N);
        mu = k_train(:,shuffledIndices(1:gmm_c)); % pick M random samples as initial mean estimates
        [~,assignedCentroidLabels] = min(pdist2(mu',k_train'),[],1); % assign each sample to the nearest mean
        for m = 1:gmm_c % use sample covariances of initial assignments as initial covariance estimates
            Sigma(:,:,m) = cov(k_train(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
            if(isnan(Sigma(:,:,m)))
                Sigma(:,:,m) = eye(d,d);
            end
        end
        t = 0; %displayProgress(t,x,alpha,mu,Sigma);

        Converged = 0; % Not converged at the beginning
        while ~Converged
            for l = 1:gmm_c
                temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(k_train,mu(:,l),Sigma(:,:,l));
            end
            plgivenx = temp./sum(temp,1);
            alphaNew = mean(plgivenx,2);
            w = plgivenx./repmat(sum(plgivenx,2),1,N);
            muNew = k_train*w';
            for l = 1:gmm_c
                v = k_train-repmat(muNew(:,l),1,N);
                u = repmat(w(l,:),d,1).*v;
                SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
            end
            Dalpha = sum(abs(alphaNew-alpha));
            Dmu = sum(sum(abs(muNew-mu)));
            DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
            Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
            alpha = alphaNew; 
            mu = muNew; 
            Sigma = SigmaNew;
            t = t+1;
        end
        %log likelihood of the current k fold iteration
        logLikelihood = sum(log(evalGMM(k_val,alpha,mu,Sigma))); 
        %store this value to compare later
        hoods(gmm_c,k) = logLikelihood;
    end
end
%The best number of components for each fold 
[~,components] = max(hoods);
%The componont number chosen the most over all 10 folds
comp_choice = max(mode(components));
    
%% Final GMM
clearvars -except data rgb pic Runs row_data col_data row col pixels regWeight delta comp_choice
for x = 1:Runs
    N = pixels;
    % Initialize the GMM to randomly selected samples
    gmm_c = comp_choice;
    d = 3;
    alpha = ones(1,gmm_c)/gmm_c;
    shuffledIndices = randperm(N);
    mu = data(:,shuffledIndices(1:gmm_c)); % pick M random samples as initial mean estimates
    [~,assignedCentroidLabels] = min(pdist2(mu',data'),[],1); % assign each sample to the nearest mean
    for m = 1:gmm_c % use sample covariances of initial assignments as initial covariance estimates
        Sigma(:,:,m) = cov(data(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
        if(isnan(Sigma(:,:,m)))
            Sigma(:,:,m) = eye(d,d);
        end
    end

    t = 0;
    Converged = 0; % Not converged at the beginning
    while ~Converged
        for l = 1:gmm_c
            temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(data,mu(:,l),Sigma(:,:,l));
        end
        plgivenx = temp./sum(temp,1);
        alphaNew = mean(plgivenx,2);
        w = plgivenx./repmat(sum(plgivenx,2),1,N);
        muNew = data*w';
        for l = 1:gmm_c
            v = data-repmat(muNew(:,l),1,N);
            u = repmat(w(l,:),d,1).*v;
            SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
        end
        Dalpha = sum(abs(alphaNew-alpha));
        Dmu = sum(sum(abs(muNew-mu)));
        DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
        Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
        alpha = alphaNew; 
        mu = muNew; 
        Sigma = SigmaNew;
        t = t+1;
    end
    alpha_total = alpha';
    mu_total = mu;
    Sigma_total = Sigma;

    results = zeros(gmm_c, pixels);
    for i=1:gmm_c
        results(i,:) = evalGMM(data,alpha_total(i),mu_total(:,i), Sigma_total(:,:,i));
    end
    [~,labels] = max(results);
      
    c1_3 = find(labels==1);
    c2_3 = find(labels==2);
    c3_3 = find(labels==3);


    m=3;
    figure(x+10)
    hold on
    plot(col_data(c1_3),-row_data(c1_3),'r.','Markersize',m)
    plot(col_data(c2_3),-row_data(c2_3),'b.','Markersize',m)
    plot(col_data(c3_3),-row_data(c3_3),'g.','Markersize',m)
    hold off
end



