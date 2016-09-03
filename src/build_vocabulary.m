% Starter code prepared by James Hays for CS 143, Brown University

%This function will sample SIFT descriptors from the training images,
%cluster them with kmeans, and then return the cluster centers.


% The inputs are images, a N x 1 cell array of image paths and the size of 
% the vocabulary.

% The output 'vocab' should be vocab_size x 128. Each row is a cluster
% centroid / visual word.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be thrown away here
  (but possibly used for extra credit in get_bags_of_sifts if you're making
  a "spatial pyramid").
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

% dΪ���������ӵ�ά��

[centers, assignments] = vl_kmeans(X, K)  
 http://www.vlfeat.org/matlab/vl_kmeans.html
  X is a d x M matrix of sampled SIFT features, where M is the number of
   features sampled. M should be pretty large! Make sure matrix is of type
   single to be safe. E.g. single(matrix).
  K is the number of clusters desired (vocab_size)
  centers is a d x K matrix of cluster centroids. This is your vocabulary.
   You can disregard 'assignments'.

  Matlab has a build in kmeans function, see 'help kmeans', but it is
  slower.
%}

% Load images from the training set. To save computation time, you don't
% necessarily need to sample from all images, although it would be better
% to do so. You can randomly sample the descriptors from each image to save
% memory and speed up the clustering. Or you can simply call vl_dsift with
% a large step size here, but a smaller step size in make_hist.m. 

% For each loaded image, get some SIFT features. You don't have to get as
% many SIFT features as you will in get_bags_of_sift.m, because you're only
% trying to get a representative sample here.

% Once you have tens of thousands of SIFT features from many training
% images, cluster them with kmeans. The resulting centroids are now your
% visual word vocabulary.


function vocab = build_vocabulary( image_paths, vocab_size )
 %%
 %    ��Ҫ�����Ĳ����� number_of_SIFT_feats��Steps
 %    ����������Ҫ��vocab.matɾ������������
 %%

 number_of_images=size(image_paths,1);   % ���ڽ����Ӿ��ʻ�ĳ�ȡͼ����Ŀ��proj3.m˵����Щͼ����trainͼ��
 number_of_SIFT_feats=100;                % ÿ��ͼ���ȡ��������Ŀ

 % ���ӳ���ͼ���л�ȡ��һ����Ŀ��������������all_SIFT_feats�����У�����K-means
 all_SIFT_feats=zeros(128,number_of_images*number_of_SIFT_feats);  

 for i=1:number_of_images
    image=im2single(imread(image_paths{i}));  % ����vl_dsift()���������ƣ������ͼ��Ҫת��Ϊsingle����
    [~, SIFT_feats] = vl_dsift(image,'Fast', 'Step', 10,'Size',8);  % SIFT_featsΪ128*������Ŀ�ľ���
    for j=1:number_of_SIFT_feats              % ֻ����number_of_SIFT_feats��Ŀ������
      all_SIFT_feats(:,(i-1)*number_of_SIFT_feats+j)=SIFT_feats(:,j);  % ��һ����Ŀ��������������all_SIFT_feats������
    end
 end
 [center,~]=vl_kmeans(all_SIFT_feats,vocab_size);  % centers:128*vocab_size
 % �������size vocab_size*128��ÿһ�б�ʾK-means��һ���������ģ�Ҳ����128ά�ռ��һ����
 vocab=center'; 
end

