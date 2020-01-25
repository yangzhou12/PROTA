function [ newfea ] = projPROTA_BCR( TX, model )
% PROTA_BCR projection with the trained model
%
% %[Syntax]%: 
%    newfea = projPROTA_BCR( TX, model)
%
% %[Inputs]%:
%    TX:            the (N+1)th-order input tensor of size d_1 x ... x d_N x numSpl
%    model:         the trained PROTA_BCR model
%
% %[Outputs]%:
%    newfea:        the projected P-dimensional features
%
% %[Toolbox needed]%:
%   This function needs the tensor toolbox v2.6 available at
%   http://www.sandia.gov/~tgkolda/TensorToolbox/

    Us = model.Us; tau = model.tau;
    Sigma = model.Sigma;
    TXmean = model.TXmean;
    P = model.P;

    N = ndims(TX)-1; % The order of samples.
    IsTX = size(TX);
    Is = IsTX(1:N); % The dimensions of the tensor
    numSpl = IsTX(N+1); % Number of samples
    
    % Centering Data
    TX = bsxfun(@minus,TX, TXmean); %Centering    
    X_vec = reshape(TX, prod(Is), numSpl); % Vectorization
    
    % Compute Projected Features
    W = khatrirao(Us,'r'); 
    WtW = ones(P);
    for n = 1:N
        expUtU = Us{n}'*Us{n} + Is(n)*Sigma{n};
        WtW = WtW.*expUtU;
    end     
    Minv = eye(P)/(tau*WtW + eye(P));
    newfea = tau*Minv*W'*X_vec;       
end

