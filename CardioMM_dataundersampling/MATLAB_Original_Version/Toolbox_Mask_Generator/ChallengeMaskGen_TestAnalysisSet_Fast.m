%% The matlab script for mask generation for TestSet and AnalysisSet - Faster when loop inside this function
% Author: Zi Wang (zi.wang@imperial.ac.uk)
% August 29, 2025

% If you want to use the code, please cite the following paper:
% [1] Zi Wang et al., CMRxRecon2024: A multimodality, multiview k-space
% dataset boosting universal machine learning for accelerated cardiac MRI, Radiology: Artificial Intelligence, 7(2): e240443, 2025.

function ChallengeMaskGen_TestAnalysisSet_Fast(data_path,mainSavePath_mask,mainSavePath_kus,file_nameCT,file_nameVD,file_nameID,dataName,setName)

%% 5D kspace [nx, ny, sc, sz, t] or 4D kspace [nx, ny, sc, sz]
var = load(data_path);  % Loading raw kspace data
if isfield(var, 'kspace_full')
    kspace_full = var.kspace_full;
elseif isfield(var, 'kspace')
    kspace_full = var.kspace;
else
    disp('No k-space data in the .mat file.');
end
nx = size(kspace_full, 1);
ny = size(kspace_full, 2);
if length(size(kspace_full)) == 5
    nt = size(kspace_full, 5);
elseif length(size(kspace_full)) == 4
    nt = 1;
else
    nt = 1;
end
ncalib = 20;

%% 2D+t Mask generation [nx, ny, nt] or [nx, ny]

% patterns = ["Uniform", "ktGaussian", "ktRadial"]; % pattern type: Uniform8, ktGaussian16, ktRadial24
% Rs = [8, 16, 24];
% for i = 1 : length(patterns)
%     pattern = patterns{i};
%     R = Rs(i);

patterns = ["Uniform", "ktGaussian", "ktRadial"]; % pattern type one-by-one: Uniform, ktGaussian, ktRadial
for i = 1 : length(patterns)
    pattern = patterns{i};
    for R = [8, 16, 24]

        mask = ktMaskGenerator(nx, ny, nt, ncalib, R, pattern);
        %% Save mask and undersampled kspace
        if strcmp(setName,'AnalysisSet') || strcmp(setName,'TestSet') || strcmp(setName,'TestAnalSet')
            savepath1 = strcat(mainSavePath_mask,'/',file_nameCT,'/',file_nameVD,'/',file_nameID);
        %     savepath2 = strcat(mainSavePath_kus,'/',file_nameCT,'/',file_nameVD,'/',file_nameID);
            mkdir(savepath1);
        %     mkdir(savepath2);
            save(fullfile(savepath1,[dataName,'_mask_',pattern,num2str(R),'.mat']),'mask','-v7.3');
        %     if strcmp(setName,'AnalysisSet') || strcmp(setName,'TestSet') || strcmp(setName,'TestAnalSet')
        %         mask_multi = reshape(mask, [nx,ny,1,1,nt]);
        %         kus = kspace_full .* mask_multi;
        %         save(fullfile(savepath2,[dataName,'_kus_',pattern,num2str(R),'.mat']),'kus','-v7.3');
        %     end
        end
    end
end

%% Mask display
% figure(R),imshow(mask,[]);
% figure(R+1),imshow(squeeze(mask(60,:,:)),[]);
end

