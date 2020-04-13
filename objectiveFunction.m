function objFncValue = objectiveFunction(X,Y,sizeParams,vecParams)
N = size(X,2); % number of samples
nX = sizeParams(1); %number of inputs
nPerceptrons = sizeParams(2); %number of perceptrons
nY = sizeParams(3); %number of outputs

params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);

H = mlpModel(X,params); %Apply the MLP to the input data to get H
objFncValue = sum((H-Y).^2)/N; %MSE function, to be minimized

end

