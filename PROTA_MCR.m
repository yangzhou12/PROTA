function model = PROTA_MCR( TX, P, varargin )
% PROTA_MCR: Probabilistic Rank-One Tensor Analysis with Moment-Based Concurrent Regularization
%
% %[Syntax]%: 
%   model = PROTA_MCR( TX, P )
%   model = PROTA_MCR( ___, Name, Value )
%
% %[Inputs]%:
%   TX:            the (N+1)th-order input tensor of size I_1 x ... x I_N x numSpl
%                  E.g., 30x20x10x100 for 100 samples of size I_1 x I_2 x I_3 = 30x20x10
%   P:             the number of extraced features
%     
% %[Name-Value Pairs]
%   'regParam':    the regularization parameter \gamma
%                  500 (Default)
%
%   'maxIters':    the maximum number of iterations
%                  500 (Default)
%
%   'tol':         the tolerance of the relative change of log-likelihood
%                  1e-5 (Default)
%
% %[Outputs]%:
%   model.Us:      the factor matrices, consisting of N matrices, one for each mode
%   model.sigma:   the noise variance
%   model.TXmean:  the mean of the training tensors TX
%   model.liklhd:  the log-likelihood at each iteration
%                        
% %[Toolbox needed]%:
%   This function needs the tensor toolbox v2.6 available at
%   http://www.sandia.gov/~tgkolda/TensorToolbox/
%                       
% %[Reference]%:            
%   Yang Zhou, Haiping Lu, Yiu-ming Cheung. 
%   Probabilistic Rank-One Tensor Analysis with Concurrent Regularization. 
%   IEEE Transactions on Cybernetics, Vol. xx, No. xx, Pages xxxx-xxxx, 2019.
%                             
% %[Author Notes]%   
%   Author:        Yang ZHOU
%   Email :        yangzhou@comp.hkbu.edu.hk
%                  youngzhkbu@gmail.com
%   Affiliation:   Department of Computer Science
%                  Hong Kong Baptist University
%   Release date:  May 01, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

if nargin < 2, error('Not enough input arguments.'); end
N = ndims(TX) - 1; % The order of samples.
IsTX = size(TX);
Is = IsTX(1:N); % The dimensions of the tensor
numSpl = IsTX(N+1); % Number of samples

ip = inputParser;
ip.addParameter('regParam', 1e3, @isscalar);
ip.addParameter('maxIters', 500, @isscalar);
ip.addParameter('tol', 1e-5, @isscalar);
ip.parse(varargin{:});

gamma  = ip.Results.regParam;
maxK   = ip.Results.maxIters; % Default Max Number of Iterations
epsCvg = ip.Results.tol; % Threshold of Convergence

%   Data Centering
TXmean = mean(TX,N+1); % The mean
TX = bsxfun(@minus,TX,TXmean);
X_vec = reshape(TX, prod(Is), numSpl);
smpVar = sum(sum(X_vec.^2));

%   Random initialization
Us = cell(N,1);
for n = 1:N
    Us{n} = normc(rand(Is(n), P));
end
sigma = sum(sum(X_vec.^2))/(numSpl*prod(Is));

%   PROTA Iterations
liklhd = zeros(maxK,1);
for iI = 1 : maxK
    %   Calc Expectation    
    W = khatrirao(Us,'r'); 
    WtW = W'*W;
    Minv = eye(P)/(WtW + sigma*eye(P));             
    expZ = Minv * W' * X_vec;        

    covZ = numSpl*sigma*Minv + expZ*expZ';
    covZ = gamma*eye(P) + covZ;
    
    %   Update U
    for n = 1:N        
        Un = zeros(Is(n),P);
        tmpIs = Is([1:n-1,n+1:N]);
        tmpUs = Us([1:n-1,n+1:N]);
        tmpTX = reshape(permute(TX,[n,[1:n-1,n+1:N],N+1]),[Is(n),prod(tmpIs),numSpl]);        
        Usmn = khatrirao(tmpUs,'r');
        for m = 1:numSpl
            Xm = tmpTX(:,:,m);
            Un = Un + Xm*Usmn*diag(expZ(:,m));
        end
        
        UsmnUsmn = WtW./(Us{n}'*Us{n});
        Un = Un / (covZ.*UsmnUsmn); 
        Us{n} = Un; 
        WtW = UsmnUsmn.*(Us{n}'*Us{n});
    end    
    
    %   Update sigma
    W = khatrirao(Us,'r');
    sigma = smpVar - sum(sum(expZ.*(W'*X_vec)));
    sigma = sigma/(numSpl*prod(Is));

    %   Calc Likelihood
    G = W*W' + sigma*eye(prod(Is)); % Total covariance matrix
    %   Calc log determinant
    det = 2 * sum(log(diag(chol(G))));
    nloglk = - 0.5*(numSpl*prod(Is)*log(2*pi) + numSpl*det + sum(sum(X_vec'/G.*X_vec',2))); 

    %   Check Convergence:    
    liklhd(iI) = nloglk;
    if iI > 1
        threshold = abs(liklhd(iI) - liklhd(iI-1)) / abs(liklhd(iI));
        if threshold < epsCvg
            liklhd = liklhd(1:iI);
            disp('Log Likelihood Converge.'); break; 
        end
        fprintf('Iteration %u, Likelihood = %f, Threshold = %f.\n', iI, liklhd(iI), threshold);
    else 
        fprintf('Iteration %u, Likelihood = %f.\n', iI, liklhd(1));
    end    
end

model.Us = Us; 
model.sigma = sigma;
model.TXmean = TXmean;
model.liklhd = liklhd;

