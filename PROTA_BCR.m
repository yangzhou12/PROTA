function [ model ] = PROTA_BCR( TX, varargin )
% PROTA_BCR: Probabilistic Rank-One Tensor Analysis with Bayesian Concurrent Regularization
%
% %[Syntax]%: 
%   model = PROTA_BCR( TX )
%   model = PROTA_BCR( ___, Name, Value )
%
% %[Inputs]%:
%   TX:            the (N+1)th-order input tensor of size I_1 x ... x I_N x numSpl
%                  E.g., 30x20x10x100 for 100 samples of size I_1 x I_2 x I_3 = 30x20x10
%     
% %[Name-Value Pairs]
%   'maxDim'       the initialized number of extraced features
%                  I_1*I_2*...*I_N - 1 (Default)
%
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
%   model.tau:     the noise precision
%   model.Sigma:   the covariance matrices, one for each U^(n)
%   model.P:       the estimated feature number
%   model.TXmean:  the mean of the training tensors TX
%   model.LB:      the variational lower bound at each iteration
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
%                  youngzhou12@gmail.com
%   Affiliation:   Department of Computer Science
%                  Hong Kong Baptist University
%   Release date:  May 01, 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

if nargin < 1, error('Not enough input arguments.'); end
N = ndims(TX) - 1; % The order of samples.
IsTX = size(TX);
Is = IsTX(1:N); % The dimensions of the tensor
numSpl = IsTX(N+1); % Number of samples

ip = inputParser;
ip.addParameter('maxDim', prod(Is) - 1, @isscalar);
ip.addParameter('regParam', var(TX(:)), @isscalar);
ip.addParameter('maxIters', 500, @isscalar);
ip.addParameter('tol', 1e-5, @isscalar);
ip.parse(varargin{:});

P          = ip.Results.maxDim; % Dimensionality of Subspace
Reg_gamma  = ip.Results.regParam;
maxIter    = ip.Results.maxIters; % Default Max Number of Iterations
epsCvg     = ip.Results.tol; % Threshold of Convergence

%   Data Centering
TXmean = mean(TX,N+1); % The mean
TX = bsxfun(@minus,TX,TXmean);
X_vec = reshape(TX, prod(Is), numSpl);

%   Random initialization
Us = cell(N,1);
for n = 1:N
    Us{n} = randn(Is(n), P);
end

tau = 1;
a_tau0 = 1e-6; b_tau0 = 1e-6;

lmbd = Reg_gamma*ones(P,1); 
unfoldX = cell(N,1); 
UsmnUsmn = cell(N,1); 
expUtU = cell(N,1); Sigma = cell(N,1);
for n = 1:N
    Sigma{n} = eye(P);
    expUtU{n} = Us{n}'*Us{n} + Is(n)*Sigma{n};
    tmpIs = Is([1:n-1,n+1:N]);
    unfoldX{n} = reshape(permute(TX,[1:n-1,n+1:N,N+1,n]),[numSpl*prod(tmpIs),Is(n)]);
end

%   PROTA_BCR Iterations
Fit = 0;
LB = zeros(maxIter,1);
for iIt = 1 : maxIter
    FitOld = Fit;
    %   Update Z    
    W = khatrirao(Us,'r'); 
    WtW = 1;
    for n = 1:N
        WtW = WtW.*expUtU{n};
    end    
    Minv = eye(P)/(tau*WtW + eye(P));             
    expZ = tau*Minv*W'*X_vec;        
    expZZt = numSpl*Minv + expZ*expZ';
    
    %   Update U
    for n = 1:N
        Lmbd = diag(lmbd);
        UsmnUsmn{n} = WtW./(expUtU{n});
        Sigma{n} = eye(P)/(tau*(expZZt+Lmbd).*UsmnUsmn{n});
        
        tmpUs = Us([1:n-1,n+1:N]);
        Usmn = khatrirao([tmpUs;expZ'],'r');
        Us{n} = (tau*Sigma{n}*Usmn'*unfoldX{n})'; 
        
        expUtU{n} = Us{n}'*Us{n} + Is(n)*Sigma{n};
        WtW = UsmnUsmn{n}.*expUtU{n};
    end        

    %   Update tau
    W = khatrirao(Us,'r');    
    trXX = X_vec.^2; trXX = sum(trXX(:));
    err = trXX - 2*sum(sum(expZ.*(W'*X_vec))) + sum(sum(expZZt.*WtW));
    
    a_tau = a_tau0 + 0.5*numSpl*prod(Is);
    b_tau = b_tau0 + 0.5*err;
    tau = a_tau / b_tau;

    %   Calc Lower Bound    
    termLklhd = -0.5*numSpl*prod(Is)*safelog(2*pi) ...
        + 0.5*numSpl*prod(Is)*(psi(a_tau)-safelog(b_tau)) - 0.5*tau*err;
    termPz = 0;
%     termPz = - numSpl*0.5*P*(1+safelog(2*pi)) - 0.5*trace(expZZt);
    termPUs =0;
%     for n = 1:N
%         termPUs = termPUs + 0.5*Is(n)*(sum(log(Reg_gamma*diag(UsmnUsmn{n}))) + ...
%             (psi(a_tau)-safelog(b_tau)) );
%     end
%     termPUs = -0.5*P*sum(Is)*safelog(2*pi) + termPUs -0.5*N*tau*Reg_gamma*trace(WtW);  
    termPTau = -safelog(gamma(a_tau0)) + a_tau0*safelog(b_tau0) ...
        + (a_tau0-1)*(psi(a_tau)-safelog(b_tau)) - b_tau0*tau;
    termQUs = 0;
%     for n = 1:N
%         termQUs = termQUs + Is(n)*0.5*logdet(Sigma{n}) + Is(n)*0.5*P*(1+safelog(2*pi));
%     end
    detMinv = 2 * sum(log(diag(chol(Minv))));
    termQz = numSpl*0.5*detMinv + numSpl*0.5*P*(1+safelog(2*pi));
    termQTau = safelog(gamma(a_tau)) - (a_tau-1)*psi(a_tau) - safelog(b_tau) + a_tau;
    LowerBound = termLklhd + termPz + termPUs + termPTau + termQUs + termQz + termQTau;
    
    %   Calc Fittness
    Fit = 1 - sqrt(err)/norm(TX(:));
    RelChan = abs(FitOld - Fit);
      
    %   Prune out Unnecessary Components
    cmpNorm = sum(W.^2);
    Uall = cell2mat(Us);
    pruneTol = sum(Is)*eps(norm(Uall,'fro'));
    numDim = sum(cmpNorm>pruneTol);
    if max(numDim) == 0
        disp('Dimensionality becomes 0 !!!');
        break;
    end
    if iIt > 1
        if P ~= max(numDim)
            dimIdx = cmpNorm > pruneTol;
            lmbd = lmbd(dimIdx);
            for n = 1:n
                Us{n} = Us{n}(:,dimIdx);
                Sigma{n} = Sigma{n}(dimIdx,dimIdx);
                expUtU{n} = expUtU{n}(dimIdx,dimIdx);
            end
            P = max(numDim);
        end
    end
    
    %   Check Convergence:    
    LB(iIt) = LowerBound;
    if iIt > 1
        threshold = abs(LB(iIt) - LB(iIt-1)) / abs(LB(iIt));
        if threshold < epsCvg && RelChan < epsCvg
            disp('Log Likelihood Converge.'); break; 
        end
        fprintf('Iter %u, ELBO = %f, LBTol = %f, Fit = %f, FitTol = %f, P = %u.\n', ...
            iIt, LB(iIt), threshold, Fit, RelChan, P);
    else 
        fprintf('Iter %u, ELBO = %f, Fit = %f, P = %u.\n', iIt, LB(1), Fit, P);
    end
end

model.Us = Us;
model.tau = tau;
model.Sigma = Sigma;
model.TXmean = TXmean;
model.LB = LB(1:iIt);
model.P = P;

function y = safelog(x)
    x(x<1e-300)=1e-200;
    x(x>1e300)=1e300;
    y=log(x);


