function  [imgs_R imgs_G imgs_B] = load_images3(paths)
            %[imgs imgsCB imgsCR] = load_images(paths)
imgs = cell(size(paths));
imgs_R = cell(size(paths));
imgs_G = cell(size(paths));
imgs_B = cell(size(paths));
for i = 1:numel(paths)
    X = imread(paths{i});
    if size(X, 3) == 3 % we extract our features from Y channel
        %X = rgb2ycbcr(X);        
        %X = rgb2gray(X);
        imgs_R{i} = im2single(X(:,:,1)); 
        imgs_G{i} = im2single(X(:,:,2)); 
        imgs_B{i} = im2single(X(:,:,3));
       
    end
    X = im2single(X); % to reduce memory usage
    imgs{i} = X;
end
