%% The generator for undersampling mask
% Author: Zi Wang (zi.wang@imperial.ac.uk)
% February 7, 2025

% If you want to use the code, please cite the following paper:
% [1] Zi Wang et al., CMRxRecon2024: A multimodality, multiview k-space
% dataset boosting universal machine learning for accelerated cardiac MRI, Radiology: Artificial Intelligence, e240443, 2025.

function mask = ktMaskGenerator(nx,ny,nt,ncalib,R,pattern)

switch pattern
    case 'Uniform'
        mask_2d = UniformSampling(nx, ny, ncalib, R);
        mask = repmat(reshape(mask_2d, [size(mask_2d, 1), size(mask_2d, 2), 1]), [1, 1, nt]);
        % realAF = numel(mask)/sum(mask(:));
        % disp(realAF);
    case 'ktUniform'
        mask = ktUniformSampling(nx, ny, nt, ncalib, R);
        % realAF = numel(mask)/sum(mask(:));
        % disp(realAF);
    case 'ktGaussian'
        alpha = 0.2;  % default: 0.28, 0<alpha<1 controls sampling density; 0: uniform density, 1: maximally non-uniform density
                      % the larger the better (stronger incoherence)
        seed = randi([1 1000]);  % default: 10, randi([1 1000]), seed to generate random numbers; a fixed seed for reproduction
        mask = ktGaussianSampling(nx, ny, nt, ncalib, R, alpha, seed);
        % realAF = numel(mask)/sum(mask(:));
        % disp(realAF);
    case 'ktRadial'
        angle4next = 137.5;  % default: 0, rotate a golden-angle for generating different mask in different time frame
        cropcorner = true;  % true or false
        R = R * 0.6;
        mask = ktRadialSampling(nx, ny, nt, ncalib, R, angle4next, cropcorner);
        % realAF = numel(mask)/sum(mask(:));
        % disp(realAF);
    otherwise
        disp('No selected undersampling pattern. Please choose the proper one.')
end