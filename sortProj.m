function [ odrIdx, stFR ] = sortProj( projFea, gnd )
%   Sorting features by data variance or Fisher scores
%   Copyright (c) Haiping LU (h.lu@sheffield.ac.uk)

    [vecDim, numSpl] = size(projFea);    
    if max(gnd)<0
        %%%%%%%%%%%%%%%%%%%%%%%%Sort by Variance%%%%%%%%%%%%%%%%%%%%%%%%
        Ymean=mean(projFea,2);%Should be zero
        projFea=projFea-repmat(Ymean,1,numSpl);
        TVars=diag(projFea*projFea');
        [~,odrIdx]=sort(TVars,'descend');
    else
        %%%%%%%%%%%%%%%Sort according to Fisher's discriminality%%%%%%%%%%%
        classLabel = unique(gnd);
        nClass = length(classLabel);%Number of classes
        ClsIdxs=cell(nClass);
        Ns=zeros(nClass,1);
        for i=1:nClass
            ClsIdxs{i}=find(gnd==classLabel(i));
            Ns(i)=length(ClsIdxs{i});
        end
        Ymean=mean(projFea,2);
        TSW=zeros(vecDim,1);
        TSB=zeros(vecDim,1);
        for i=1:nClass
            clsYp=projFea(:,ClsIdxs{i});
            clsMean=mean(clsYp,2);
            FtrDiff=clsYp-repmat(clsMean,1,Ns(i));
            TSW=TSW+sum(FtrDiff.*FtrDiff,2);
            meanDiff=clsMean-Ymean;
            TSB=TSB+Ns(i)*meanDiff.*meanDiff;
        end
        FisherRatio=real(TSB./TSW);
        [stFR,odrIdx]=sort(FisherRatio,'descend');
    end
end

