%% The matlab script for mask generation (CMRxRecon MICCAI2025)
% Author: Zi Wang (zi.wang@imperial.ac.uk)
% February 20, 2025

% If you want to use the code, please cite the following paper:
% [1] Zi Wang et al., CMRxRecon2024: A multimodality, multiview k-space
% dataset boosting universal machine learning for accelerated cardiac MRI, Radiology: Artificial Intelligence, 7(2): e240443, 2025.

clc
clear all
close all
warning off
%% Add path
currentFolder = pwd;
addpath(genpath(currentFolder));
%% Mask generation
for R = [8, 16, 24]
    % 5D kspace [kx, ky, sc, sz, t] or 4D kspace [kx, ky, sc, sz]
    kspace = ones(256, 255, 10, 10, 12);
    nx = size(kspace, 1);
    ny = size(kspace, 2);
    nt = size(kspace, 5);
    ncalib = 20;
    pattern = 'Uniform';  % Uniform, ktGaussian, ktRadial
    
    % Mask generation [kx, ky, t] or [kx, ky]
    mask = ktMaskGenerator(nx, ny, nt, ncalib, R, pattern);
    save(fullfile([pattern,'_R',num2str(R),'.mat']),'mask','-v7.3');
    
    % Mask display
    figure(R),imshow(mask(:,:,1),[]);
    figure(R+1),imshow(squeeze(mask(60,:,:)),[]);
end
