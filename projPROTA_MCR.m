function [ newfea ] = projPROTA_MCR( TX, model)
% PROTA_MCR projection with the trained model
%
% %[Syntax]%: 
%    newfea = projPROTA_MCR( TX, model)
%
% %[Inputs]%:
%    TX:            the (N+1)th-order input tensor of size d_1 x ... x d_N x numSpl
%    model:         the trained PROTA_MCR model
%
% %[Outputs]%:
%    newfea:        the projected P-dimensional features
%
% %[Toolbox needed]%:
%   This function needs the tensor toolbox v2.6 available at
%   http://www.sandia.gov/~tgkolda/TensorToolbox/

    Us = model.Us; sigma = model.sigma;
    TXmean = model.TXmean;

    N = ndims(TX)-1; % The order of samples.
    IsTX = size(TX);
    Is = IsTX(1:N); % The dimensions of the tensor
    numSpl = IsTX(N+1); % Number of samples
    P = size(Us{1},2); 
    
    % Centering Data
    TX = bsxfun(@minus,TX, TXmean); %Centering    
    X_vec = reshape(TX, prod(Is), numSpl); % Vectorization
    
    % Compute Projected Features
    W = khatrirao(Us,'r'); 
    WtW = W'*W;
    M = WtW + sigma*eye(P);
    newfea = M \ W' * X_vec;                   
end

