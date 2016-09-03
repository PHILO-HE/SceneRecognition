% Starter code prepared by James Hays for CS 143, Brown University

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters.


% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation. % d 为聚类后的中心点的个数
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
  category. This is useful for creating the binary labels for each SVM
  training task.

[W B] = vl_svmtrain(features, labels, LAMBDA)
  http://www.vlfeat.org/matlab/vl_svmtrain.html

  This function trains linear svms based on training examples, binary
  labels (-1 or 1), and LAMBDA which regularizes the linear classifier
  by encouraging W to be of small magnitude. LAMBDA is a very important
  parameter! You might need to experiment with a wide range of values for
  LAMBDA, e.g. 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.

  Matlab has a built in SVM, see 'help svmtrain', which is more general,
  but it obfuscates the learned SVM parameters in the case of the linear
  model. This makes it hard to compute "confidences" which are needed for
  one-vs-all classification.

%}

%unique() is used to get the category list from the observed training
%category list. 'categories' will not be in the same order as in proj3.m, 
%because unique() sorts them. This shouldn't really matter, though.


function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
%%
%     需要调整的参数：LAMBDA
%%
  LAMBDA=0.001;
  
  test_image_number=size(test_image_feats,1);   % test图像特征描述的数目，也等于test图像的数目
  % train_image_number=size(train_image_feats,1);  % train图像的数目
  
  categories=unique(train_labels);   % 每个train图像所属的类别
  number_of_categories=size(categories,1);  % 类别的数目
 
  % 用于存放vl_svmtrain的输出参数W，行数与一个特征描述的维度相同，列数与类别数目相同
  vl_W=zeros(size(train_image_feats,2),number_of_categories);  
  
  % 用于存放vl_svmtrain的输出参数b，第一列的某一行是vl_svmtrain的一次输出，行数和类别的数目相同，列数与test图像的数目相同
  % 第二列之后（包括第二列）为第一列的重复
  B=zeros(number_of_categories,test_image_number);  
  
  for i=1:number_of_categories
     Y=ones(size(train_labels,1),1).*(-1);   % 1-vs-all，对某个类别，属于该类别的与train_image_feats'索引位置相对应的
     Y(strcmp(train_labels,categories(i)))=1;  % 类别为categories(i)的标记为1，其余位置仍为-1
     
     % svm求解+1、-1区分的train_image_feats，用Y=sign(W*X+b)对这两个类划分,得到W,b
     [vl_W(:,i),b] = vl_svmtrain(train_image_feats', Y, LAMBDA);   
     B(i,1)=b;
  end
  W=vl_W';  % vl_W的行数与一个特征描述的维度相同，列数为类别的数目，W与之相反
  for i=2:test_image_number
     B(:,i)=B(:,1);
  end
  
  % 矩阵Y_TEST描述各个类别下（1-vs-all）的W、B参数对每个test图像的结果
  Y_TEST=zeros(number_of_categories,test_image_number);
  for i=1:test_image_number
      Y_TEST=W*test_image_feats'+B;
  end
  
  % 将Y_TEST排序，看哪个类别下（1-vs-all）的参数得到的值最大
  % 最大的值所在的类别作为test的预测类别
  [~,index]=sort(Y_TEST,1,'descend');
  predicted_categories=cell(test_image_number,1);
  for i=1:test_image_number
    predicted_categories(i,1)=categories(index(1,i),1);
  end
end



% Y_TEST=X*W+B的代码，不用转置

% function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
% 
%   %参数LAMBDA需要调整
%   LAMBDA=0.01;
% 
%   test_image_number=size(test_image_feats,1);   % test图像的数目
%   train_image_number=size(train_image_feats,1);  % train图像的数目
% 
%   categories=unique(train_labels);   % train_labels(train图像)所属的类别
%   number_of_categories=size(categories,1);  % 类别的数目
%  
%   % 为vl_svmtrain的输出权重，行数与一个特征描述的维度相同，列数与类别数目相同
%   vl_W=zeros(size(train_image_feats,2),number_of_categories);  
%   % 第一列的某一行是vl_svmtrain的一次输出，行数和类别的数目相同，列数与test图像的列数相同
%   % 第二列之后（包括第二列）为第一列的重复
%   B=zeros(test_image_number,number_of_categories);  
%   
%   for i=1:number_of_categories
%      Y=ones(size(train_labels,1),1).*(-1);   % 1-vs-all，对某个类别，属于该类别的与train_image_feats'索引位置相对应的
%      Y(strcmp(train_labels,categories(i)))=1;  % Y上，标记为1，其余位置标记为-1
%      
%      % svm求解+1、-1区分的train_image_feats，Y=sign(W*X+b)描述,得到W,b
%      [vl_W(:,i),b] = vl_svmtrain(train_image_feats', Y, LAMBDA);   
%      B(1,i)=b;
%   end
%   W=vl_W';  % vl_W的行数与一个特征描述的维度相同，列数为类别的数目，W与之相反
%   for i=2:test_image_number
%      B(i,:)=B(1,:);
%   end
%   
%   % 矩阵Y_TEST描述各个类别下（1-vs-all）的W、B参数对每个test图像的结果
%   %Y_TEST=zeros(train_image_number,test_image_number);
%   Y_TEST=zeros(test_image_number,number_of_categories);
%   for i=1:test_image_number
%       Y_TEST=test_image_feats*vl_W+B;
%   end
%   % 将Y_TEST排序，看哪个类别下（1-vs-all）的参数得到的值最大
%   % 最大的值所在的类别作为test的预测类别
%   [~,index]=sort(Y_TEST,2,'descend');
%   
%   predicted_categories=cell(test_image_number,1);
%   for i=1:test_image_number
%     predicted_categories(i,1)=categories(index(i,1),1);
%   end
% end













