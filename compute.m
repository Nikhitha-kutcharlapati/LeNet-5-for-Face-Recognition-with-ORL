imgs = [];
for i=1:size(trainlist, 1)
    imgpath = table2array(trainlist(i,1));
    img = im2double(imread(char(imgpath)));
    imgs = [imgs img];
end
mean = mean2(imgs);
std = std2(imgs);