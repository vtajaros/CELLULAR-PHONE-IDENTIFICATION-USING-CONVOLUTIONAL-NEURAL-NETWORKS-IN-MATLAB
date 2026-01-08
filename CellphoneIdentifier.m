function CellphoneIdentifier
    % Create the figure
    hFig = figure('Position', [100, 100, 600, 400], 'MenuBar', 'none', ...
                  'Name', 'Cellphone Identifier Using CNN', 'NumberTitle', 'off', ...
                  'Resize', 'off');

    % Create an axes to display the image
    hAxes = axes('Parent', hFig, 'Units', 'pixels', ...
                 'Position', [50, 100, 500, 200]);

    % Create a push button to load the image
    hButton = uicontrol('Parent', hFig, 'Style', 'pushbutton', ...
                        'String', 'Load Image', ...
                        'Position', [250, 20, 100, 40], ...
                        'Callback', @loadImageCallback);
    
    % Create a text label to display the classification result
    hText = uicontrol('Parent', hFig, 'Style', 'text', ...
                      'Position', [50, 360, 1500, 20], ...
                      'HorizontalAlignment', 'left', ...
                      'FontSize', 9, 'BackgroundColor', 'white');

    function loadImageCallback(~, ~)
        % Open a file selection dialog
        [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'; ...
                                          '*.*', 'All Files (*.*)'}, ...
                                          'Select an Image');
        % Check if a file was selected
        if isequal(filename, 0)
            return; % User canceled the file selection
        end
        
        % Read and display the image
        img = imread(fullfile(pathname, filename));
        imshow(img, 'Parent', hAxes);
        
        % Classify the image using your CNN
        [result, accuracy] = classifyImageWithCNN(img);
        
        % Display the classification result
        resultStr = sprintf('Classification Result: %s\nAccuracy: %.2f%%', result, accuracy*100);
        set(hText, 'String', resultStr);
    end
end

function [result, accuracy] = classifyImageWithCNN(img)  
    outputFolder = fullfile('phonedata');
    rootFolder = fullfile(outputFolder);

    categories = {'blackberrycingular', 'infinixnote12', ...
        'iphone12', 'iphone15', 'samsunga4', 'samsunga20', 'samsunga50', 'samsungkeystone'};

    %image data store
    imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');

    tbl = countEachLabel(imds);
    minSetCount = min(tbl{:,2});

    imds = splitEachLabel(imds, minSetCount, 'randomize');
    countEachLabel(imds);

    % blackberrycingular = find(imds.Labels == 'blackberrycingular', 1);
    % infinixnote12 = find(imds.Labels == 'infinixnote12', 1);
    % iphone12 = find(imds.Labels == 'iphone12', 1);
    % iphone15 = find(imds.Labels == 'iphone15', 1);
    % samsunga4 = find(imds.Labels == 'samsunga4', 1);
    % samsunga20 = find(imds.Labels == 'samsunga20', 1);
    % samsunga50 = find(imds.Labels == 'samsunga50', 1);
    % samsungakeystone = find(imds.Labels == 'samsungakeystone', 1);

    net = resnet50();

    net.Layers(1);
    net.Layers(end);

    %training
    numel(net.Layers(end).ClassNames);
    [trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

    %resizing
    imageSize = net.Layers(1).InputSize;

    augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');

    augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb'); 

    %convolution
    % w1 = net.Layers(2).Weights;
    % w1 = mat2gray(w1);

    featureLayer = 'fc1000';
    trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

    trainingLables = trainingSet.Labels;
    classifier = fitcecoc(trainingFeatures, trainingLables, 'Learner', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

    testFeatures = activations(net, augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

    predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns'); %predicted labels

    testLables = testSet.Labels; %actual labels

    %confusion matrix
    confMat = confusionmat(testLables, predictLabels);
    confMat = bsxfun(@rdivide, confMat, sum(confMat,2));

    accuracy = mean(diag(confMat)); %accuracy

    ds = augmentedImageDatastore(imageSize, img, 'ColorPreprocessing', 'gray2rgb'); 

    imageFeatures = activations(net, ds, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

    result = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');
end
