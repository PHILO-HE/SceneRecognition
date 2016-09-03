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
%     ��Ҫ�����Ĳ�����Step
%%
 load('vocab.mat')                                % proj3.m ���б���vocab�Ĵ���
 vocab_size = size(vocab, 1);                     % vocabÿһ�б�ʾK-means�õ���һ�����ģ�����vocab_size=size(vocab,2)

 number_of_images=size(image_paths,1);            % ͼ�����Ŀ
 image_feats=zeros(number_of_images,vocab_size);  % �����SIFT����������Ϊnumber_of_image*vocal_sizeά�� 

 display('runing for geting bags of sifts');
 for i=1:number_of_images
    image=im2single(imread(image_paths{i}));         % ����vl_dsift���ƣ���Ҫת��Ϊsingle����
    [~, SIFT_feats] = vl_dsift(image, 'Step', 5,'Size',6);   % SIFT_features is a 128 x N matrix of SIFT features
    
    SIFT_feats=single(SIFT_feats);    % ����vl_alldist2()���ƣ���Ҫת��Ϊsingle����
    vocab=single(vocab);              % ����vl_alldist2()���ƣ���Ҫת��Ϊsingle����
    dist=vl_alldist2(vocab',SIFT_feats);
 
    [~,index]=sort(dist,1); % ŷʽ����������Ӿ��ʻ㣬���Ա���Ϊ��ͼ���г���
    number_of_SIFT_feats=size(SIFT_feats,2);  % vl_dsift�����SIFT_featsΪ128*SIFT��Ŀ
    for j=1:number_of_SIFT_feats
        image_feats(i,index(1,j))=image_feats(i,index(1,j))+1;   % ͳ��ֱ��ͼ��Ҳ���������Ӿ��ʻ���ͼ���г��ֵĴ���
    end
    image_feats(i,index(1,j))=image_feats(i,index(1,j))./sum(image_feats(i,:));  % ֱ��ͼ��һ��
 end
end
