Train = imageDatastore("C:\Users\marty\Documents\studia\biocybernetyka\projekt\baza danych\dane\uczący", 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
%0 - zbiór łagodny, 1 - zbiór złośliwy
Validation = imageDatastore("C:\Users\marty\Documents\studia\biocybernetyka\projekt\baza danych\dane\walidacyjny", 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
Test = imageDatastore("C:\Users\marty\Documents\studia\biocybernetyka\projekt\baza danych\dane\testowy", 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
tbl = countEachLabel(Train); %zliczenie ilości zdjęć w zbiorach

imageAugmenter = imageDataAugmenter('RandRotation', [-20,20], 'RandXTranslation', [-3,3], 'RandYTranslation', [-3,3])
imageSize = [224 224 3];
augimds = augmentedImageDatastore(imageSize, Train, 'DataAugmentation',imageAugmenter)

layers = [
    imageInputLayer(imageSize) %warstwa wejściowa (można dodać normalizacje)
    
    convolution2dLayer(3,4,'Padding','same') %warstwa splotowa z przesuwanymi filtrami(8 filtrów 3x3) (Padding - dopełnienie krawędzi wejściowej - taki sam rozmiar wejscia i wyjscia)
    %opcja z dodaniem parametrow dla wag/odchylenia
    batchNormalizationLayer %warstwa normalizująca minipartie. Przyspiesza uczenie i zmniejsza wrażliwość
    reluLayer %operacja progowa. wartość mniejsza od zera ustawiana jest na 0
    
    maxPooling2dLayer(2,'Stride',2) %próbkowanie w dół. dzieli dane wejścio Rwe na prostokąty i oblicza maksimum każdego (rozmiar puli [2 2] z krokiem [3 3]
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    
    fullyConnectedLayer(2) %mnoży dane wejściowe przez macierz wag i dodaje wektor odchylenia
    softmaxLayer %funkcja aktywacji softmax - zmienia wartości rzeczywiste w prawdopodobieństwa
    classificationLayer]; %oblicza utrate entropii

options = trainingOptions('adam', 'LearnRateSchedule','piecewise', 'LearnRateDropFactor',0.8 , 'LearnRateDropPeriod',5, 'ValidationData', Validation,'MaxEpochs', 100, 'MiniBatchSize',64, 'Plots','training-progress', 'Shuffle', 'once','ValidationPatience',3, 'OutputNetwork','last-iteration')
%LearnRateDropFactor i LearnRateDropPeriod
net = trainNetwork(augimds,layers,options);



TestPred = classify(net, Test);
PredTest = Test.Labels;

accuracyTest = sum(TestPred == PredTest)/numel(PredTest)

[C1, order1] = confusionmat(Test.Labels, TestPred)
figure, confusionchart(C1)
title('Macierz pomyłek zbioru testowego')


ValidationPred = classify(net, Validation);
PredValidation = Validation.Labels;

accuracyValidation = sum(ValidationPred == PredValidation)/numel(PredValidation)

[C2, order2] = confusionmat(Validation.Labels, ValidationPred)
figure, confusionchart(C2)
title('Macierz pomyłek zbioru walidacyjnego')


TrainPred = classify(net, Train);
PredTrain = Train.Labels;

accuracy = sum(TrainPred == PredTrain)/numel(PredTrain)

[C3, order3] = confusionmat(Train.Labels, TrainPred)
figure, confusionchart(C3)
title('Macierz pomyłek zbioru treningowego')


save('ccc.mat', 'net')