% Starter code prepared by James Hays for CS 143, Brown University

%This feature representation is described in the handout, lecture
%materials, and Szeliski chapter 14.


% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x 128
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every time at significant expense.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram.

% You will want to construct SIFT features here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% SIFT features will look very different from a smaller version of the same
% image.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be used for extra
  credit if you are constructing a "spatial pyramid".
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

D = vl_alldist2(X,Y) 
   http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '  You can use this to figure out the closest
    cluster center for every SIFT feature. You could easily code this
    yourself, but vl_alldist2 tends to be much faster.

Or:

For speed, you might want to play with a KD-tree algorithm (we found it
reduced computation time modestly.) vl_feat includes functions for building
and using KD-trees.
 http://www.vlfeat.org/matlab/vl_kdtreebuild.html

%}

function image_feats = get_bags_of_sifts(image_paths)
%%
%     需要调整的参数：Step
%%
 load('vocab.mat')                                % proj3.m 中有保存vocab的代码
 vocab_size = size(vocab, 1);                     % vocab每一行表示K-means得到的一个中心，不是vocab_size=size(vocab,2)

 number_of_images=size(image_paths,1);            % 图像的数目
 image_feats=zeros(number_of_images,vocab_size);  % 输出的SIFT特征描述包为number_of_image*vocal_size维度 

 display('runing for geting bags of sifts');
 for i=1:number_of_images
    image=im2single(imread(image_paths{i}));         % 由于vl_dsift限制，需要转换为single类型
    [~, SIFT_feats] = vl_dsift(image, 'Step', 5,'Size',6);   % SIFT_features is a 128 x N matrix of SIFT features
    
    SIFT_feats=single(SIFT_feats);    % 由于vl_alldist2()限制，需要转换为single类型
    vocab=single(vocab);              % 由于vl_alldist2()限制，需要转换为single类型
    dist=vl_alldist2(vocab',SIFT_feats);
 
    [~,index]=sort(dist,1); % 欧式距离最近的视觉词汇，可以被认为在图像中出现
    number_of_SIFT_feats=size(SIFT_feats,2);  % vl_dsift的输出SIFT_feats为128*SIFT数目
    for j=1:number_of_SIFT_feats
        image_feats(i,index(1,j))=image_feats(i,index(1,j))+1;   % 统计直方图，也就是所有视觉词汇在图像中出现的次数
    end
    image_feats(i,index(1,j))=image_feats(i,index(1,j))./sum(image_feats(i,:));  % 直方图归一化
 end
end
