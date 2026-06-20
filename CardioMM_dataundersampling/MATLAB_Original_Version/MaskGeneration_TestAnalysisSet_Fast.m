%% The matlab script for varied mask generation for TestSet and AnalysisSet - Faster when loop inside
% Author: Zi Wang (zi.wang@imperial.ac.uk)
% August 29, 2025

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

%% Set path
TaskName = 'TaskAll';  % TaskAll
setName = 'TestSet';  % TestSet, AnalysisSet

% TestSet or AnalysisSet
basePath = '/mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil';
SavebasePath = basePath;

mainDataPath = strcat(basePath,'/');
FileList = dir(mainDataPath);  % Different modalities
FileList = FileList(~ismember({FileList.name}, {'.', '..'}));
NumFile = length(FileList);

%% Running all modalities, datasets, samplings, and patient IDs
for ind0 = 1 : NumFile
    file_name = FileList(ind0).name;
    if contains(file_name,'')
        disp(['Running modality file:',file_name]);
        dataPath = strcat(mainDataPath,file_name,'/');  % Example: '.../MultiCoil/Cine/'
        fileNameSet = dir(dataPath);
        fileNameSet = fileNameSet(~ismember({fileNameSet.name}, {'.', '..'}));
        fileSetLen = length(fileNameSet);  % Example: if only have TrainingSet, fileSetLen = 1; if have three type of sets, fileSetLen = 3.
        for ind1 = 1 : fileSetLen
            file_nameSet = fileNameSet(ind1).name;
            disp(['Running dataset file:',file_nameSet]);
            dataPathSet = strcat(dataPath,file_nameSet,'/');  % Example: '.../MultiCoil/Cine/TestSet/'
            fileNameFS = dir(dataPathSet);
            fileNameFS = fileNameFS(~ismember({fileNameFS.name}, {'.', '..'}));
            fileFSLen = length(fileNameFS);  % Example: if only have FullSample, fileFSLen = 1; if have N type of sampling, fileFSLen = N.
            for ind2 = 1 : fileFSLen
                file_nameFS = fileNameFS(ind2).name;
                if contains(file_nameFS,'FullSample')
                    disp(['Running sampling file:',file_nameFS]);
                    dataPathFS = strcat(dataPathSet,file_nameFS,'/');  % Example: '.../MultiCoil/Cine/TestSet/FullSample/'
                    fileNameCT = dir(dataPathFS);
                    fileNameCT = fileNameCT(~ismember({fileNameCT.name}, {'.', '..'}));
                    fileCTLen = length(fileNameCT);  % Example: if only have Center001, fileCTLen = 1; if have N different centers, fileCTLen = N.
                    for ind22 = 1 : fileCTLen
                        file_nameCT = fileNameCT(ind22).name;
                        disp(['Running center file:',file_nameCT]);
                        dataPathCT = strcat(dataPathFS,file_nameCT,'/');  % Example: '.../MultiCoil/Cine/TestSet/FullSample/Center1/'
                        fileNameVD = dir(dataPathCT);
                        fileNameVD = fileNameVD(~ismember({fileNameVD.name}, {'.', '..'}));
                        fileVDLen = length(fileNameVD);  % Example: if only have Siemens_30T_Vida, fileVDLen = 1; if have N different vendors, fileVDLen = N.
                        for ind222 = 1 : fileVDLen
                            file_nameVD = fileNameVD(ind222).name;
                            disp(['Running vendor file:',file_nameVD]);
                            dataPathVD = strcat(dataPathCT,file_nameVD,'/');  % Example: '.../MultiCoil/Cine/TestSet/FullSample/Center001/Siemens_30T_Vida/'
                            fileNameID = dir(dataPathVD);
                            fileNameID = fileNameID(~ismember({fileNameID.name}, {'.', '..'}));
                            fileIDLen = length(fileNameID);  % Example: if only have P001, fileIDLen = 1; if have N different patient ID, fileIDLen = N.
                            for ind3 = 1 : fileIDLen
                                file_nameID = fileNameID(ind3).name;
                                disp(['Running ID file:',file_nameID]);
                                dataPathID = strcat(dataPathVD,file_nameID,'/');  % Example: '.../MultiCoil/Cine/TestSet/FullSample/Center001/Siemens_30T_Vida/P001/'
                                fileNamemat = dir(dataPathID);
                                fileNamemat = fileNamemat(~ismember({fileNamemat.name}, {'.', '..'}));
                                filematLen = length(fileNamemat);  % Example: if only have cine_sax.mat, filematLen = 1; if have N different .mat, filematLen = N.
                                % Running all .mat
                                for ind4 = 1 : filematLen
                                    imgName = fileNamemat(ind4).name;
                                    
                                    %% Different .mat files
                                    if contains(imgName,'.mat') && ~contains(imgName,'mask') && ~contains(imgName,'kus')
                                        dataName = strrep(imgName, '.mat', '');  % replace .mat to none, get dataName
                                        data_path = strcat(dataPathID,imgName);  % Path for loading .mat
                                        mainSavePath_mask = strcat(SavebasePath,'/',file_name,'/',setName,'/',['Mask_',TaskName],'/');  % Main path for saving generated masks
                                        mainSavePath_kus = strcat(SavebasePath,'/',file_name,'/',setName,'/',['UnderSample_',TaskName],'/');  % Main path for saving generated undersampled kspace
                                        % Parameters for mask generation
                                        if strcmp(file_nameSet,setName)
                                            ChallengeMaskGen_TestAnalysisSet_Fast(data_path,mainSavePath_mask,mainSavePath_kus,file_nameCT,file_nameVD,file_nameID,dataName,file_nameSet);
                                            % Save example: '.../MultiCoil/Cine/TestSet/Mask_TaskAll/Center001/Siemens_30T_Vida/P001/cine_sax_mask_Uniform8.mat'
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
