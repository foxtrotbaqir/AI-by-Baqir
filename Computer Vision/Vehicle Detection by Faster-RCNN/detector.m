% Detector code
% this code is tested on image03 folder image sequence from video taken out
% of KITTI vision benchmark suite. So make sure to either change the
% directory of the imageset in the filepath variable or download the mentioned folder. 

%% gathering label data from loaded gTruth variable
load('scenelabels.mat')
[imds , blds] = objectDetectorTrainingData(gTruth);
cds = combine(imds , blds);
%% Creating a new faster R-CNN based on transfer learning.
inputImageSize = [512 1392 3]; % size of input image
numClasses = 5; % number of objects to detect
load('NeoNet')  % enter network
network = NeoNet;
numAnchors = numClasses;
anchorBoxes = estimateAnchorBoxes(cds,numAnchors);
featureLayer =  'relu13'; %choose any activation layer of the loaded network
lgraph = fasterRCNNLayers(inputImageSize,numClasses,anchorBoxes,network, featureLayer);
analyzeNetwork(lgraph)
%% Train newly created faster RCNN detector
% creating training options
options = trainingOptions('sgdm',...
      'MiniBatchSize', 16, ...
      'InitialLearnRate', 1e-2, ...
      'LearnRateSchedule','piecewise',...
      'LearnRateDropPeriod', 2,...
      'LearnRateDropFactor', 0.5,...
      'MaxEpochs', 4, ...
      'VerboseFrequency', 50, ...
      'L2Regularization',0.0005,...
      'Momentum',0.9,...
      'ExecutionEnvironment','auto',...
      'CheckpointPath', tempdir); 
net = trainFasterRCNNObjectDetector(cds, lg, options); 
%% Use the detector on a loaded image. Store the locations of the bounding boxes and their detection scores.
figure;
hold on
for i = 1:length(files)
    I = readimage(imds, i);
    [bboxes,scores,labels] = detect(net,I);
    % Annotate the image with the detections and their scores.
    I = insertObjectAnnotation(I,'rectangle',bboxes,labels);
    imshow(I)
    title('Detected Vehicles and Detected Labels')
end
