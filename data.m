clear variables; close all; clc;

trainf = fopen('./trainlist.txt', 'w');
testf = fopen('./testlist.txt', 'w');
for i=1:40
    folder = ['./data/ORL/s' num2str(i) '/'];
    filelist = dir(fullfile(folder, '*.bmp'));
    randIndex = randperm(10);
    for j=1:8
        filedir = fullfile(folder, filelist(randIndex(j)).name);
        fprintf(trainf, '%s %d\n', filedir, i-1);
    end
    for k=9:10
        filedir = fullfile(folder, filelist(randIndex(k)).name);
        fprintf(testf, '%s %d\n', filedir, i-1);
    end
end
fclose(trainf);
fclose(testf);