function H = mlpModel(X,params)
N = size(X,2);                          
nY = length(params.d);                  
U = params.A*X + repmat(params.b,1,N);  
Z = activationFunction(U);              
V = params.C*Z + repmat(params.d,1,N); 
H = V;
end

