

%clear all; close all; clc;  
warning off all   
p = pwd;
%addpath(fullfile(p, '/methods'));  % the upscaling methods
addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm
addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm
addpath(fullfile(p, '/include'))


imgscale = 1;   % the scale reference we work with

ressss = cell(1,3);
 upscaling = [2]; % the magnification factor
input_dir = 'testset';
img_dir = dir(fullfile(input_dir, '*.bmp'));
%input_dir = 'Set10TMM'; % Directory with input images

% pattern = '*.bmp'; % Pattern to process
pattern = '*.bmp'; % Pattern to process

dict_sizes = [2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536];
neighbors = [1:1:12, 16:4:32, 40:8:64, 80:16:128, 256, 512, 1024];

%nn_patch_max = 1;
%d = 7
%for nn=1:28
%nn= 28

num_patches_cluster = 5000000;


disp(['The experiment uses ' input_dir ' dataset and aims at a magnification of factor x' num2str(upscaling) '.']);


fprintf('\n\n');
num_centers_use = 6
    
 d = 10   %1024
   
   
    idx_nn_patch_max = 11;
    nn_patch_max = 2^idx_nn_patch_max;
    disp(['nn_patch_max = ' num2str(nn_patch_max)]);
    
    tag = [input_dir '_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms'];
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%  Dictionary learning via K-means clustering  %%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    conf.filenames = glob(input_dir, pattern); % Cell array      
    
        conf.scale = upscaling; % scale-up factor
        conf.level = 1; % # of scale-ups to perform
        conf.window = [3 3]; % low-res. window size
        conf.border = [1 1]; % border of the image (to ignore)
         conf.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf.filters = {G, G.', L, L.'}; % 2D versions
        conf.interpolate_kernel = 'bicubic';
          conf.overlap = [1 1]; % partial overlap (for faster training)
  
  
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%   MY_ASC        %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda=0.03;;tao=0.001;
lambda_m=0.03;tao_m=0.001;
decimal=4;
[imgs_R imgs_G imgs_B]=load_images3(glob(['rgbTra2_gnd_x' num2str(upscaling) '_' num2str(lambda*1000)], '*.bmp'));%load_imgs(path)
 
[imgs_Rt imgs_Gt imgs_Bt]=load_images3(glob(['rgbTra2_my_x' num2str(upscaling) '_' num2str(lambda*1000)], '*.bmp')); 
%%%%%%%%%  rrrrrrrrrrrrrrrrrr   %%%%%%%%%%%%%%%%%%%%%%%%%
 
 mat_file = ['T3_conf_KmeansR_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling) '_l' num2str( lambda_m*1000) '_t' num2str(tao*1000)];   

 if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'T3_conf_KmeansR');

    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using K-means approach...']);
        % Simulation settings
        conf_KmeansR.scale = upscaling; % scale-up factor
        conf_KmeansR.level = 1; % # of scale-ups to perform
        conf_KmeansR.window = [3 3]; % low-res. window size
        conf_KmeansR.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf_KmeansR.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf_KmeansR.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf_KmeansR.filters = {G, G.', L, L.'}; % 2D versions
        conf_KmeansR.interpolate_kernel = 'bicubic';

        conf_KmeansR.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf_KmeansR.overlap = [1 1]; % partial overlap (for faster training)
        end
        
        startt = tic;
       
        conf_KmeansR= cluster_kmeans_pp_T(conf_KmeansR, imgs_R,imgs_Rt ,dict_sizes(d));       
        conf_KmeansR.overlap = conf_KmeansR.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf_KmeansR.trainingtime = toc(startt);
        toc(startt)       
        T3_conf_KmeansR=conf_KmeansR;
        save(mat_file, 'T3_conf_KmeansR');
        % train call       
 end
 
  
 
%%%%%%%%%  ggggggggggg  %%%%%%%%%%%%%%%%%%%%%%%%%
 mat_file = ['T3_conf_KmeansG_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling) '_l' num2str( lambda_m*1000) '_t' num2str(tao*1000)];   

 if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'T3_conf_KmeansG');

    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using K-means approach...']);
        % Simulation settings
        conf_KmeansG.scale = upscaling; % scale-up factor
        conf_KmeansG.level = 1; % # of scale-ups to perform
        conf_KmeansG.window = [3 3]; % low-res. window size
        conf_KmeansG.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf_KmeansG.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf_KmeansG.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf_KmeansG.filters = {G, G.', L, L.'}; % 2D versions
        conf_KmeansG.interpolate_kernel = 'bicubic';

        conf_KmeansG.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf_KmeansG.overlap = [1 1]; % partial overlap (for faster training)
        end
        
        startt = tic;
       
        conf_KmeansG= cluster_kmeans_pp_T(conf_KmeansG, imgs_G ,imgs_Gt ,dict_sizes(d));       
        conf_KmeansG.overlap = conf_KmeansG.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf_KmeansG.trainingtime = toc(startt);
        toc(startt)       
        T3_conf_KmeansG=conf_KmeansG;
        save(mat_file, 'T3_conf_KmeansG');
        % train call        
 end
  
%%%%%%%%%  bbbbbbbbbbbbbbbbb   %%%%%%%%%%%%%%%%%%%%%%%%%
 mat_file = ['T3_conf_KmeansB_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling) '_l' num2str( lambda_m*1000) '_t' num2str(tao*1000)];   

 if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'T3_conf_KmeansB');

    else                            
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using K-means approach...']);
        % Simulation settings
        conf_KmeansB.scale = upscaling; % scale-up factor
        conf_KmeansB.level = 1; % # of scale-ups to perform
        conf_KmeansB.window = [3 3]; % low-res. window size
        conf_KmeansB.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf_KmeansB.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf_KmeansB.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf_KmeansB.filters = {G, G.', L, L.'}; % 2D versions
        conf_KmeansB.interpolate_kernel = 'bicubic';

        conf_KmeansB.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf_KmeansB.overlap = [1 1]; % partial overlap (for faster training)
        end
        
        startt = tic;
       
        conf_KmeansB= cluster_kmeans_pp_T(conf_KmeansB, imgs_B  ,imgs_Bt, dict_sizes(d));       
        conf_KmeansB.overlap = conf_KmeansB.window - [1 1]; % full overlap scheme (for better reconstruction)    
        conf_KmeansB.trainingtime = toc(startt);
        toc(startt)   
        T3_conf_KmeansB=conf_KmeansB;
        save(mat_file, 'T3_conf_KmeansB');
        % train call        
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     
    %num_centers_use = 30;       % num_centers_use nearest centers to use 
    %nn_patch_max = 2048;        % max number of the nn patches to each cluster center
    
    fname = ['T3_MY_x' num2str(upscaling) '_K' num2str(dict_sizes(d)) '_max' num2str(nn_patch_max) '_d' num2str(num_centers_use) '_lambda' num2str( lambda_m*1000000) '_' num2str(decimal) 't' num2str(tao_m*1000000) '.mat'];
    %fnameS = ['MYs_x' num2str(upscaling) '_K' num2str(dict_sizes(d)) '_max' num2str(nn_patch_max) '_d' num2str(num_centers_use) '_lambda' num2str( lambda_m*1000) '_t'  num2str(tao_m*1000) '.mat'];
    
    %fname = ['kmeans_sub_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(nn_patch_max) 'nn_' num2str(num_centers_use) 'sub_whole.mat'];
    % e.g. kmeans_sub_x3_1024atoms_2048nn_40sub_whole.mat
    if exist(fname,'file')
       load(fname);
       %load(fnameS);
    else
        %%
       disp('Compute MY_T3 regressors');
       ttime = tic;
       tic
      
       [imgs_R imgs_G imgs_B]=load_images3(glob(['rgbTra2_gnd_x' num2str(upscaling) '_' num2str(lambda*1000)], '*.bmp'));
       [imgs_Rt imgs_Gt imgs_Bt]=load_images3(glob(['rgbTra2_my_x' num2str(upscaling) '_' num2str(lambda*1000)], '*.bmp')); 

       [plores_r phires_r] = collectSamplesScales_T(T3_conf_KmeansR,imgs_R,imgs_Rt, 12, 0.98);  
       [plores_g phires_g] = collectSamplesScales_T(T3_conf_KmeansG,imgs_G,imgs_Gt, 12, 0.98);  
       [plores_b phires_b] = collectSamplesScales_T(T3_conf_KmeansB,imgs_B,imgs_Bt, 12, 0.98);  

%         if size(plores,2) > num_patches_custer                
%             plores = plores(:,1:num_patches_cluster);
%             phires = phires(:,1:num_patches_cluster);
%         end
%%%%%%%%%%% rrrrrrrrrrrrr  %%%%%%%%%%%%%%

        number_samples = size(plores_r,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores_r.^2).^0.5+eps;
        l2n = repmat(l2,size(plores_r,1),1);    
        l2(l2<0.1) = 1;
        plores_r = plores_r./l2n;
        clear l2n
        l2n_h = repmat(l2,size(phires_r,1),1);
        clear l2
        phires_r = phires_r./l2n_h;
        clear l2n_h
    
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores_g.^2).^0.5+eps;
        l2n = repmat(l2,size(plores_g,1),1);    
        l2(l2<0.1) = 1;
        plores_g= plores_g./l2n;
        clear l2n
        l2n_h = repmat(l2,size(phires_g,1),1);
        clear l2
        phires_g = phires_g./l2n_h;
        clear l2n_h
% l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores_b.^2).^0.5+eps;
        l2n = repmat(l2,size(plores_b,1),1);    
        l2(l2<0.1) = 1;
        plores_b = plores_b./l2n;
        clear l2n
        l2n_h = repmat(l2,size(phires_b,1),1);
        clear l2
        phires_b = phires_b./l2n_h;
        clear l2n_h

        %llambda_kmeans_sub = 0.1;
        %cluster the whole data with kmeans plus plus
        %llambda_kmeans_sub = 0.1;
        %cluster the whole data with kmeans plus plus
        folder_current = pwd;
        run([folder_current, '\vlfeat-0.9.19\toolbox\vl_setup.m']);
        tic;
        [centers_r, assignments_r] = vl_kmeans(plores_r, dict_sizes(10), 'Initialization', 'plusplus');
        assignments_r = double(assignments_r);
        [centers_g, assignments_g] = vl_kmeans(plores_g, dict_sizes(10), 'Initialization', 'plusplus');
        assignments_g = double(assignments_g);
        [centers_b, assignments_b] = vl_kmeans(plores_b, dict_sizes(10), 'Initialization', 'plusplus');
        assignments_b = double(assignments_b);
  disp('done cluster!')
  toc;
       
        %load('cluster_x3_whole.mat');
    
        CluP3 = [];%CluP_S=[];
         P_Dic=[zeros(size(phires_r,1)) eye(size(phires_r,1)) zeros(size(phires_r,1));...
                     zeros(size(phires_r,1)) zeros(size(phires_r,1)) eye(size(phires_r,1));...
                     eye(size(phires_r,1)) zeros(size(phires_r,1)) zeros(size(phires_r,1))]; 
       
        
        for i = 1:1024%size(conf.dict_lores, 2)
            tic;
            Dr = pdist2(single(centers_r'), single(T3_conf_KmeansR.dict_lore_kmeans(:, i)'));
            [~, idx_centersR] = sort(Dr, 'ascend');
            
            idx_centers_use = idx_centersR(1:num_centers_use);
            idx_patch_use = [];
            for i_temp = 1:num_centers_use
                idx_temp = find(assignments_r == idx_centers_use(i_temp));
                idx_patch_use = [idx_patch_use idx_temp];
            end
            sub_plores = plores_r(:, idx_patch_use);
            sub_phires = phires_r(:, idx_patch_use);
            
            if nn_patch_max <= length(idx_patch_use)
                sub_Dr = pdist2(single(sub_plores'), single(T3_conf_KmeansR.dict_lore_kmeans(:, i)'));
                [~, sub_idx] = sort(sub_Dr, 'ascend');
                LoR = sub_plores(:, sub_idx(1:nn_patch_max));
                HiR = sub_phires(:, sub_idx(1:nn_patch_max));
            else
                LoR = sub_plores;
                HiR = sub_phires;
            end
            
           %%%%%%%%%%%%%%%%%%% 
  
            Dg = pdist2(single(centers_g'), single(T3_conf_KmeansG.dict_lore_kmeans(:, i)'));
            [~, idx_centersG] = sort(Dg, 'ascend');
            
            idx_centers_use = idx_centersG(1:num_centers_use);
            idx_patch_use = [];
            for i_temp = 1:num_centers_use
                idx_temp = find(assignments_g == idx_centers_use(i_temp));
                idx_patch_use = [idx_patch_use idx_temp];
            end
            sub_plores = plores_g(:, idx_patch_use);
            sub_phires = phires_g(:, idx_patch_use);
            
            if nn_patch_max <= length(idx_patch_use)
                sub_Dg = pdist2(single(sub_plores'), single(T3_conf_KmeansG.dict_lore_kmeans(:, i)'));
                [~, sub_idx] = sort(sub_Dg, 'ascend');
                LoG = sub_plores(:, sub_idx(1:nn_patch_max));
                HiG = sub_phires(:, sub_idx(1:nn_patch_max));
            else
                LoG = sub_plores;
                HiG = sub_phires;
            end

            Db = pdist2(single(centers_b'), single(T3_conf_KmeansB.dict_lore_kmeans(:, i)'));
            [~, idx_centersB] = sort(Db, 'ascend');
            
            idx_centers_use = idx_centersB(1:num_centers_use);
            idx_patch_use = [];
            for i_temp = 1:num_centers_use
                idx_temp = find(assignments_b == idx_centers_use(i_temp));
                idx_patch_use = [idx_patch_use idx_temp];
            end
            sub_plores = plores_b(:, idx_patch_use);
            sub_phires = phires_b(:, idx_patch_use);
            
            if nn_patch_max <= length(idx_patch_use)
                sub_Db = pdist2(single(sub_plores'), single(T3_conf_KmeansB.dict_lore_kmeans(:, i)'));
                [~, sub_idx] = sort(sub_Db, 'ascend');
                LoB = sub_plores(:, sub_idx(1:nn_patch_max));
                HiB = sub_phires(:, sub_idx(1:nn_patch_max));
            else
                LoB = sub_plores;
                HiB = sub_phires;
            end
         
            
  DH=[HiR zeros(size(HiR,1),size(HiG,2)) zeros(size(HiR,1),size(HiB,2));...
           zeros(size(HiG,1),size(HiR,2)) HiG zeros(size(HiG,1),size(HiB,2));...
           zeros(size(HiB,1),size(HiR,2)) zeros(size(HiB,1),size(HiG,2)) HiB ];
 
  DL=[LoR zeros(size(LoR,1),size(LoG,2)) zeros(size(LoR,1),size(LoB,2));...
      zeros(size(LoG,1),size(LoR,2)) LoG zeros(size(LoG,1),size(LoB,2));...
      zeros(size(LoB,1),size(LoR,2)) zeros(size(LoB,1),size(LoG,2)) LoB ];
  
  P_sc=[zeros(size(LoR,2),size(LoR,2)) eye(size(LoR,2),size(LoG,2)) zeros(size(LoR,2),size(LoB,2));...
            zeros(size(LoG,2),size(LoR,2)) zeros(size(LoG,2),size(LoG,2)) eye(size(LoG,2),size(LoB,2));...
            eye(size(LoB,2),size(LoR,2)) zeros(size(LoB,2),size(LoG,2)) zeros(size(LoB,2),size(LoB,2))];
      
  CluP3{i}= DH*pinv(DL'*DL+2*tao_m*DH'*DH-tao_m*P_sc'*DH'*P_Dic'*DH-...
        tao_m*DH'*P_Dic*DH*P_sc+lambda_m*eye(size(P_sc)))*DL';
    
  %CluP_S{i}= SH*pinv(DL'*DL+2*tao_m*SH'*SH-tao_m*P_sc'*SH'*P_Dic'*SH-...
        %tao_m*SH'*P_Dic*SH*P_sc+lambda_m*eye(size(P_sc)))*DL';          
    disp(['done : ' num2str(i) '/1024']);
    toc;
        end
        
     
        clear plores_r
        clear phires_r
    
        
         clear plores_g
        clear phires_g
        
        
         clear plores_b
        clear phires_b
        clear sub_plores
        clear sub_phires
        
        ttime = toc(ttime);        
        save(fname,'CluP3','ttime', 'number_samples');   
        %save(fnameS,'CluP_S','ttime', 'number_samples'); 
        toc

%}
    %% 
    end
result_dirRGB = qmkdir(['3_layer_x-' sprintf('%s_x%d-', input_dir, upscaling) 'k' num2str(dict_sizes(d)) '_max' num2str(nn_patch_max) '_lam' num2str(lambda_m*1000) '_t'  num2str(tao_m*1000) ]);
     %%
    %t = cputime;    
  
 comYUV_Tr3=[];  
    for i = 1:numel(conf.filenames)
         f = conf.filenames{i};
        [p, n, x] = fileparts(f);
        im=(imread(f));
        img1{1}=im(:,:,1);img2{1}=im(:,:,2);img3{1}=im(:,:,3);
        img1=modcrop(img1,conf.scale^conf.level);
        img2=modcrop(img2,conf.scale^conf.level);
        img3=modcrop(img3,conf.scale^conf.level);
        im = (cat(3,img1{1},img2{1},img3{1}));
 
        [imgR, imgG, imgB] = load_images3({f}); 
 
        if imgscale<1
    
            imgR = resize(imgR, imgscale, conf.interpolate_kernel);
            imgG= resize(imgG, imgscale, conf.interpolate_kernel);
            imgB = resize(imgB, imgscale, conf.interpolate_kernel);
        end
        sz = size(imgR{1});
        
        fprintf('%d/%d\t"%s" [%d x %d]\n', i, numel(conf.filenames), f, sz(1), sz(2));
        
        imgR = modcrop(imgR, conf.scale^conf.level);
        imgG= modcrop(imgG, conf.scale^conf.level);
        imgB = modcrop(imgB, conf.scale^conf.level);
      
 
            lowR = resize(imgR, 1/conf.scale^conf.level, conf.interpolate_kernel);

                lowG = resize(imgG, 1/conf.scale^conf.level, conf.interpolate_kernel);
                lowB = resize(imgB, 1/conf.scale^conf.level, conf.interpolate_kernel);
          
           %{ 

         interpolatedR = resize(lowR, conf.scale^conf.level, conf.interpolate_kernel);
          
            interpolatedG = resize(lowG, conf.scale^conf.level, conf.interpolate_kernel);
            interpolatedB = resize(lowB, conf.scale^conf.level, conf.interpolate_kernel);
  %}
       
            fprintf('MY\n');
             Dic.MYr = T1_CluPr;Dic.MYg = T1_CluPg;Dic.MYb = T1_CluPb;
            
            startt = tic;
            res_T1= scaleup_MY_T1(Dic,T1_conf_KmeansR,T1_conf_KmeansG,T1_conf_KmeansB, lowR,lowG,lowB);
            lowRt{1}=res_T1{1}(:,:,1);
            lowGt{1}=res_T1{1}(:,:,2);
            lowBt{1}=res_T1{1}(:,:,3);
            
            Dic.MY = CluP2;
            res_T2= scaleup_MY_Tm(Dic,T2_conf_KmeansR,T2_conf_KmeansG,T2_conf_KmeansB, lowRt,lowGt,lowBt);
            lowRt{1}=res_T2{1}(:,:,1);
            lowGt{1}=res_T2{1}(:,:,2);
            lowBt{1}=res_T2{1}(:,:,3);
          
            Dic.MY = CluP3;
             res= scaleup_MY_sr(Dic,T3_conf_KmeansR,T3_conf_KmeansG,T3_conf_KmeansB, lowRt,lowGt,lowBt);
            
            toc(startt)
     %imgres=res{1};
     imgres = (res{1});
     %imshow(imgres);imshow(im)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
      result = shave( imgres, conf.border * conf.scale);
 
      resultim = shave(im, conf.border * conf.scale);
      
     % imwrite(resultim,[result_dirRGB  '/gnd'  img_dir(i).name]);

    % imwrite(resultim,[result_gnd  '/gnd'  img_dir(i).name]);  
     imwrite(result,[result_dirRGB  '/my'  img_dir(i).name]);  
   
       sp_rmse = compute_rmse(resultim, result);

      sp_psnr = 20*log10(255/sp_rmse);

       sp_ssim = ssim(resultim, result);
       comYUV_Tr3=[comYUV_Tr3;sp_psnr; sp_ssim; sp_rmse;];
     
        %conf.filenames{i} = f;
    end
    save resyuv_Tr3 comYUV_Tr3;