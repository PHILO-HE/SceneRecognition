% Starter code prepared by James Hays for CS 143, Brown University

%This function will predict the category for every test image by finding
%the training image with most similar features. Instead of 1 nearest
%neighbor, you can vote based on k nearest neighbors which will increase
%performance (although you need to pick a reasonable value for k).


% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
   This can tell you which indices in train_labels match a particular
   category. Not necessary for simple one nearest neighbor classifier.

 D = vl_alldist2(X,Y) 
    http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator ' 
   vl_alldist2 supports different distance metrics which can influence
   performance significantly. The default distance, L2, is fine for images.
   CHI2 tends to work well for histograms.
 
  [Y,I] = MIN(X) if you're only doing 1 nearest neighbor, or
  [Y,I] = SORT(X) if you're going to be reasoning about many nearest
  neighbors 

%}

function predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
%%
%     需要调整的参数：K
%%
  K=30;    % K-NN算法中要考察的neighbors个数
  test_image_number=size(test_image_feats,1);  % test图像的特征描述的数目，与test图像的数目相同
  
  categories=unique(train_labels);  % 所有train图像的类别，由于train_label是对所有的train图像的描述，要找到没有重复的所有类别
  number_of_categories=size(categories,1);  % 类别的数目
  predicted_categories=cell(test_image_number,1);   % 对test图像的分类结果，作为输出
  
  % 所有训练图像与测试图像所提取的特征的欧式距离，
  % dist矩阵某一行表示某个train图像的features与所有的test图像的features欧式距离
  dist=vl_alldist2(train_image_feats',test_image_feats');   % vl_alldist2按列处理，要转置  
  
  [~,index]=sort(dist,1);                % 将dist矩阵对每列按照升序排列，index为test图像的索引
  K_labels=cell(test_image_number,K);    % 用于标记对每个测试图像，与其“距离”最近的K个train图像的label（按行表示）
  categories_statistic=zeros(test_image_number,number_of_categories);    % 用于统计这K个train图像所属的分类（按行表示）
  
   for i=1:test_image_number
     for j=1:K
      %top_K_labels(i,j)=labels(strcmp(labels, train_labels(index(j,i))));
      K_labels(i,j)=train_labels(index(j,i));    % 得到第i个test图像，距离最近的K个图像的label
     end
     for j=1:number_of_categories
       categories_statistic(i,j) = sum(strcmp(categories(j), K_labels(i,:)));  % 统计对每个类别，K个图像中有多少个属于该类别
     end
   end
   [~,index_sort]=sort(categories_statistic,2,'descend');  % 将对类别的统计结果每行按照降序排列
  
   % 对每个test图像，特征描述空间与其最近的K个图像最多属于某个类别，则认为该test图像也属于该类别
   for i=1:test_image_number
    predicted_categories(i,1)=categories(index_sort(i,1));  
   end
end







