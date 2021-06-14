function [imgmy, midresR,midresG,midresB] = scaleup_MY_Tr(Dic,conf_KmeansR,conf_KmeansG,conf_KmeansB, lowR,lowG,lowB)

% Super-Resolution Iteration
    fprintf('Scale-Up MY');
    
       %{
    midresR = resize(lowR, sz, conf_KmeansR.interpolate_kernel);    
    midresG = resize(lowG, sz, conf_KmeansG.interpolate_kernel);
    midresB = resize(lowB, sz, conf_KmeansB.interpolate_kernel);
    %}
    midresR = resize(lowR, conf_KmeansR.upsample_factor, conf_KmeansR.interpolate_kernel);    
    midresG = resize(lowG, conf_KmeansG.upsample_factor, conf_KmeansG.interpolate_kernel);
    midresB = resize(lowB, conf_KmeansB.upsample_factor, conf_KmeansB.interpolate_kernel);
    %}
    for i = 1:numel(midresR)
        featuresR = collect(conf_KmeansR, {midresR{i}}, conf_KmeansR.upsample_factor, conf_KmeansR.filters);
        featuresR = double(featuresR);
    end
     for i = 1:numel(midresG)
        featuresG = collect(conf_KmeansG, {midresG{i}}, conf_KmeansG.upsample_factor, conf_KmeansG.filters);
        featuresG = double(featuresG);
     end
      for i = 1:numel(midresB)
        featuresB = collect(conf_KmeansB, {midresB{i}}, conf_KmeansB.upsample_factor, conf_KmeansB.filters);
        featuresB = double(featuresB);
      end
        % Reconstruct using patches' dictionary and their anchored
        % projections
          
        featuresR = conf_KmeansR.V_pca'*featuresR;
        featuresG = conf_KmeansG.V_pca'*featuresG;
        featuresB = conf_KmeansB.V_pca'*featuresB;
        
        patchesR = zeros(size(Dic.MYr{1},1),size(featuresR,2));
        patchesG = zeros(size(Dic.MYg{1},1),size(featuresG,2));
        patchesB = zeros(size(Dic.MYb{1},1),size(featuresB,2));
        Dr = pdist2(single(conf_KmeansR.dict_lore_kmeans)', single(featuresR)');
        [valr idxr] = min(Dr);
        Dg = pdist2(single(conf_KmeansG.dict_lore_kmeans)', single(featuresG)');
        [valg idxg] = min(Dg);
        Db = pdist2(single(conf_KmeansB.dict_lore_kmeans)', single(featuresB)');
        [valb idxb] = min(Db);
        for l = 1:size(featuresR,2)            
            patchesR(:,l) = Dic.MYr{idxr(l)} * featuresR(:,l);
                  
            patchesG(:,l) = Dic.MYg{idxg(l)} * featuresG(:,l);
                   
            patchesB(:,l) = Dic.MYb{idxb(l)} * featuresB(:,l);
            
        end
        
        % Add low frequencies to each reconstructed patch        
        patchesR = patchesR + collect(conf_KmeansR, {midresR{i}}, conf_KmeansR.scale, {});
        patchesG = patchesG + collect(conf_KmeansG, {midresG{i}}, conf_KmeansG.scale, {});
        patchesB = patchesB + collect(conf_KmeansB, {midresB{i}}, conf_KmeansB.scale, {});
        % Combine all patches into one image
        img_size = size(lowR{i}) * conf_KmeansR.scale;%size(lowR{i}) * conf_KmeansR.scale;
        grid = sampling_grid(img_size, ...
            conf_KmeansR.window, conf_KmeansR.overlap, conf_KmeansR.border, conf_KmeansR.scale);
        resultR = overlap_add(patchesR, img_size, grid);
        resultG = overlap_add(patchesG, img_size, grid);
        resultB = overlap_add(patchesB, img_size, grid);
        resultR=uint8(resultR* 255);
        resultG=uint8(resultG* 255);
        resultB=uint8(resultB* 255);
        imgf(:,:,1)=resultR;
        imgf(:,:,2)=resultG;
        imgf(:,:,3)=resultB;
        imgmy{i}=imgf;
       clear imgf
        %imgs{i} = result; % for the next iteration
        fprintf('.');
    end

