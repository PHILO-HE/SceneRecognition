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
%     ��Ҫ�����Ĳ�����K
%%
  K=30;    % K-NN�㷨��Ҫ�����neighbors����
  test_image_number=size(test_image_feats,1);  % testͼ���������������Ŀ����testͼ�����Ŀ��ͬ
  
  categories=unique(train_labels);  % ����trainͼ����������train_label�Ƕ����е�trainͼ���������Ҫ�ҵ�û���ظ����������
  number_of_categories=size(categories,1);  % ������Ŀ
  predicted_categories=cell(test_image_number,1);   % ��testͼ��ķ���������Ϊ���
  
  % ����ѵ��ͼ�������ͼ������ȡ��������ŷʽ���룬
  % dist����ĳһ�б�ʾĳ��trainͼ���features�����е�testͼ���featuresŷʽ����
  dist=vl_alldist2(train_image_feats',test_image_feats');   % vl_alldist2���д���Ҫת��  
  
  [~,index]=sort(dist,1);                % ��dist�����ÿ�а����������У�indexΪtestͼ�������
  K_labels=cell(test_image_number,K);    % ���ڱ�Ƕ�ÿ������ͼ�����䡰���롱�����K��trainͼ���label�����б�ʾ��
  categories_statistic=zeros(test_image_number,number_of_categories);    % ����ͳ����K��trainͼ�������ķ��ࣨ���б�ʾ��
  
   for i=1:test_image_number
     for j=1:K
      %top_K_labels(i,j)=labels(strcmp(labels, train_labels(index(j,i))));
      K_labels(i,j)=train_labels(index(j,i));    % �õ���i��testͼ�񣬾��������K��ͼ���label
     end
     for j=1:number_of_categories
       categories_statistic(i,j) = sum(strcmp(categories(j), K_labels(i,:)));  % ͳ�ƶ�ÿ�����K��ͼ�����ж��ٸ����ڸ����
     end
   end
   [~,index_sort]=sort(categories_statistic,2,'descend');  % ��������ͳ�ƽ��ÿ�а��ս�������
  
   % ��ÿ��testͼ�����������ռ����������K��ͼ���������ĳ���������Ϊ��testͼ��Ҳ���ڸ����
   for i=1:test_image_number
    predicted_categories(i,1)=categories(index_sort(i,1));  
   end
end







