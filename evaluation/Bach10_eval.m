function [sdr, sir, sar] = Bach10_eval(dataDir, estimatesDir)

%% Returns SDR, SIR, SAR ratios for all the separated files
% Parameters:
%     dataDir - Base directory for Bach10. Assumes that this directory
%               contains a subdirectory for each of the songs
%     estimatesDir - Directory containing all the reconstructed sources
% Outputs:
%     sdr, sir, sar - Arrays of size (numSources, numSongs)

sourcesNames = {'bassoon','clarinet','saxphone','violin'};
nSongs = 10;
nSources = 4;
sdr = zeros(nSources, nSongs);
sir = zeros(nSources, nSongs);
sar = zeros(nSources, nSongs);
addpath('./bss_eval');

% Loop over all songs
for i=1:nSongs
   sourceData = [];
   estimateData = [];
   subDirs = dir(strcat(dataDir, sprintf('%02d', i), '*'));
   songDirName = subDirs(1).name;
   for j=1:nSources
       sourceFile = fullfile(dataDir, songDirName, [songDirName, '-', sourcesNames{j}, '.wav']);
       [audioData, fs] = audioread(sourceFile);
       sourceData(j, :) = audioData;
       
       estimateFile = fullfile(estimatesDir, [songDirName, '-', sourcesNames{j}, '.wav']);
       estimateData(j, :) = audioread(estimateFile);
   end
   [SDR, SIR, SAR] = bss_eval_sources(estimateData,sourceData);
   sdr(:, i) = SDR;
   sir(:, i) = SIR;
   sar(:, i) = SAR;
end

