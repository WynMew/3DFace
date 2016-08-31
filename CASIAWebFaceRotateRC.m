%% CASIA webFace Rotate
%% Wyn Mew @ FRDC
%%
addpath calib
datapath='/home/miaoqianwen/WorkSpace/CASIAWebFace/CASIA-WebFace';
%% 
% five rendering views:
model=cell(1,5);
model{1}=load ('renderer/model3D_BFM02_0.mat');
% model{2}=load ('renderer/model3D_BFM02_m40.mat');
% model{3}=load ('renderer/model3D_BFM02_m75.mat');
% model{4}=load ('renderer/model3D_BFM02_p40.mat');
% model{5}=load ('renderer/model3D_BFM02_p75.mat');
Model3D=model{1}.model3D;
                
subj=dir(datapath);
for m=3:length(subj)   
    ims = dir([datapath filesep subj(m).name filesep '*.jpg']);
    disp(subj(m).name);
    for i=1:length(ims) 
        [pathstr,name,ext]=fileparts(ims(i).name);
        ptsfile=[datapath filesep subj(m).name filesep name '.pts'];
        if exist(ptsfile)
            %disp ('yes');
            fidu_XY = zeros(68,2);
            pts = textread(ptsfile, '%s', 'delimiter', '\n','whitespace' , ''); %#ok<DTXTRD>
            if length(pts)==73
                for j=5:(length(pts)-1)
                    line=pts{j};
                    fidu_XY(j-4,:) = str2num(line);
                end

                [C_Q, A,R,T] = estimateCamera(Model3D, fidu_XY);
    %%           
                I_Q=imread([datapath filesep subj(m).name filesep ims(i).name]);
                % render new view
                ACC_CONST = 800; 
                I_Q = double(I_Q);
                refU = Model3D.refU;
                % back ground mask:
                bgind = sum(abs(refU),3)==0; 
                % count the number of times each pixel in the query is accessed
                threedee = reshape(refU,[],3)';
                tmp_proj = C_Q * [threedee;ones(1,size(threedee,2))];
                tmp_proj2 = tmp_proj(1:2,:)./ repmat(tmp_proj(3,:),2,1);

                bad = min(tmp_proj2)<1 | tmp_proj2(2,:)>size(I_Q,1) | tmp_proj2(1,:)>size(I_Q,2) | bgind(:)';
                tmp_proj2(:,bad) = [];
                ind = sub2ind([size(I_Q,1),size(I_Q,2)], round(tmp_proj2(2,:)),round(tmp_proj2(1,:)));
                synth_frontal_acc = zeros(size(refU,1),size(refU,2));
                ind_frontal = 1:(size(refU,1)*size(refU,2));
                ind_frontal(bad) = [];

                [c,~,ic] = unique(ind);
                count = hist(ind,c);
                synth_frontal_acc(ind_frontal) = count(ic);
                synth_frontal_acc(bgind) = 0;
                synth_frontal_acc = imfilter(synth_frontal_acc,fspecial('gaussian', 16, 30),'same','replicate');

                % create synthetic view, without symmetry
                if ndims(I_Q)==3
                    c1 = I_Q(:,:,1); f1 = zeros(size(synth_frontal_acc));
                    c2 = I_Q(:,:,2); f2 = zeros(size(synth_frontal_acc));
                    c3 = I_Q(:,:,3); f3 = zeros(size(synth_frontal_acc));
                else
                    disp('gray image');
                    I_Q=reshape(repmat(I_Q,[1,3]),[size(I_Q,1) size(I_Q,2) 3]);
                    c1 = I_Q(:,:,1); f1 = zeros(size(synth_frontal_acc));
                    c2 = I_Q(:,:,2); f2 = zeros(size(synth_frontal_acc));
                    c3 = I_Q(:,:,3); f3 = zeros(size(synth_frontal_acc));
                end

                f1(ind_frontal) = interp2(c1, tmp_proj2(1,:), tmp_proj2(2,:), 'cubic'); 
                f2(ind_frontal) = interp2(c2, tmp_proj2(1,:), tmp_proj2(2,:), 'cubic'); 
                f3(ind_frontal) = interp2(c3, tmp_proj2(1,:), tmp_proj2(2,:), 'cubic'); 
                frontal_raw = cat(3,f1,f2,f3);
                %%
                midcolumn = round(size(refU,2)/2);
                % sum along column
                sumaccs = sum(synth_frontal_acc);
                sum_left = sum(sumaccs(1:midcolumn));
                sum_right = sum(sumaccs(midcolumn+1:end));
                sum_diff = sum_left - sum_right;

                if abs(sum_diff)>ACC_CONST % one side is occluded
                    if sum_diff > ACC_CONST % left side of face has more occlusions
                        weights = [zeros(size(refU,1),midcolumn), ones(size(refU,1),midcolumn)];
                    else % right side of face has occlusions
                        weights = [ones(size(refU,1),midcolumn), zeros(size(refU,1),midcolumn)];
                    end
                    weights = imfilter(weights, fspecial('gaussian', 33, 60.5),'same','replicate');

                    % apply soft symmetry to use whatever parts are visible in ocluded
                    % side
                    synth_frontal_acc = synth_frontal_acc./max(synth_frontal_acc(:));

                    weight_take_from_org = 1./exp(0.5+synth_frontal_acc);%
                    weight_take_from_sym = 1-weight_take_from_org;

                    weight_take_from_org = weight_take_from_org.*fliplr(weights);
                    weight_take_from_sym = weight_take_from_sym.*fliplr(weights);

                    weight_take_from_org = repmat(weight_take_from_org,[1,1,3]);
                    weight_take_from_sym = repmat(weight_take_from_sym,[1,1,3]);
                    weights = repmat(weights,[1,1,3]);

                    denominator = weights + weight_take_from_org + weight_take_from_sym;
                    frontal_sym = (frontal_raw.*weights + frontal_raw.*weight_take_from_org + ...
                    flipdim(frontal_raw,2).*weight_take_from_sym)./denominator;

                    % Exclude eyes from symmetry        
                    % frontal_sym = frontal_sym.*(1-eyemask) + frontal_raw.*eyemask;

                else %% both sides are occluded pretty much to the same extent -- do not use symmetry
                    disp('do no sym');
                    frontal_sym = uint8(frontal_raw);
                end
                frontal_sym = uint8(frontal_sym);
                frontal_raw = uint8(frontal_raw);

                %imshow(frontal_raw);
                rotateimgname=[datapath filesep subj(m).name filesep name '_00.png'];
                imwrite(frontal_raw, rotateimgname); 
                rotateimgname=[datapath filesep subj(m).name filesep name '_01.png'];
                imwrite(frontal_sym, rotateimgname);      
            end
        else
            disp (['Landmark fail: ', subj(m).name filesep, ims(i).name ]) ;
        end
    end
end
disp('done!');