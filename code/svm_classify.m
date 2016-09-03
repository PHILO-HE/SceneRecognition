% Starter code prepared by James Hays for CS 143, Brown University

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters.


% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation. % d Ϊ���������ĵ�ĸ���
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
%     ��Ҫ�����Ĳ�����LAMBDA
%%
  LAMBDA=0.001;
  
  test_image_number=size(test_image_feats,1);   % testͼ��������������Ŀ��Ҳ����testͼ�����Ŀ
  % train_image_number=size(train_image_feats,1);  % trainͼ�����Ŀ
  
  categories=unique(train_labels);   % ÿ��trainͼ�����������
  number_of_categories=size(categories,1);  % ������Ŀ
 
  % ���ڴ��vl_svmtrain���������W��������һ������������ά����ͬ�������������Ŀ��ͬ
  vl_W=zeros(size(train_image_feats,2),number_of_categories);  
  
  % ���ڴ��vl_svmtrain���������b����һ�е�ĳһ����vl_svmtrain��һ�������������������Ŀ��ͬ��������testͼ�����Ŀ��ͬ
  % �ڶ���֮�󣨰����ڶ��У�Ϊ��һ�е��ظ�
  B=zeros(number_of_categories,test_image_number);  
  
  for i=1:number_of_categories
     Y=ones(size(train_labels,1),1).*(-1);   % 1-vs-all����ĳ��������ڸ�������train_image_feats'����λ�����Ӧ��
     Y(strcmp(train_labels,categories(i)))=1;  % ���Ϊcategories(i)�ı��Ϊ1������λ����Ϊ-1
     
     % svm���+1��-1���ֵ�train_image_feats����Y=sign(W*X+b)���������໮��,�õ�W,b
     [vl_W(:,i),b] = vl_svmtrain(train_image_feats', Y, LAMBDA);   
     B(i,1)=b;
  end
  W=vl_W';  % vl_W��������һ������������ά����ͬ������Ϊ������Ŀ��W��֮�෴
  for i=2:test_image_number
     B(:,i)=B(:,1);
  end
  
  % ����Y_TEST������������£�1-vs-all����W��B������ÿ��testͼ��Ľ��
  Y_TEST=zeros(number_of_categories,test_image_number);
  for i=1:test_image_number
      Y_TEST=W*test_image_feats'+B;
  end
  
  % ��Y_TEST���򣬿��ĸ�����£�1-vs-all���Ĳ����õ���ֵ���
  % ����ֵ���ڵ������Ϊtest��Ԥ�����
  [~,index]=sort(Y_TEST,1,'descend');
  predicted_categories=cell(test_image_number,1);
  for i=1:test_image_number
    predicted_categories(i,1)=categories(index(1,i),1);
  end
end



% Y_TEST=X*W+B�Ĵ��룬����ת��

% function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
% 
%   %����LAMBDA��Ҫ����
%   LAMBDA=0.01;
% 
%   test_image_number=size(test_image_feats,1);   % testͼ�����Ŀ
%   train_image_number=size(train_image_feats,1);  % trainͼ�����Ŀ
% 
%   categories=unique(train_labels);   % train_labels(trainͼ��)���������
%   number_of_categories=size(categories,1);  % ������Ŀ
%  
%   % Ϊvl_svmtrain�����Ȩ�أ�������һ������������ά����ͬ�������������Ŀ��ͬ
%   vl_W=zeros(size(train_image_feats,2),number_of_categories);  
%   % ��һ�е�ĳһ����vl_svmtrain��һ�������������������Ŀ��ͬ��������testͼ���������ͬ
%   % �ڶ���֮�󣨰����ڶ��У�Ϊ��һ�е��ظ�
%   B=zeros(test_image_number,number_of_categories);  
%   
%   for i=1:number_of_categories
%      Y=ones(size(train_labels,1),1).*(-1);   % 1-vs-all����ĳ��������ڸ�������train_image_feats'����λ�����Ӧ��
%      Y(strcmp(train_labels,categories(i)))=1;  % Y�ϣ����Ϊ1������λ�ñ��Ϊ-1
%      
%      % svm���+1��-1���ֵ�train_image_feats��Y=sign(W*X+b)����,�õ�W,b
%      [vl_W(:,i),b] = vl_svmtrain(train_image_feats', Y, LAMBDA);   
%      B(1,i)=b;
%   end
%   W=vl_W';  % vl_W��������һ������������ά����ͬ������Ϊ������Ŀ��W��֮�෴
%   for i=2:test_image_number
%      B(i,:)=B(1,:);
%   end
%   
%   % ����Y_TEST������������£�1-vs-all����W��B������ÿ��testͼ��Ľ��
%   %Y_TEST=zeros(train_image_number,test_image_number);
%   Y_TEST=zeros(test_image_number,number_of_categories);
%   for i=1:test_image_number
%       Y_TEST=test_image_feats*vl_W+B;
%   end
%   % ��Y_TEST���򣬿��ĸ�����£�1-vs-all���Ĳ����õ���ֵ���
%   % ����ֵ���ڵ������Ϊtest��Ԥ�����
%   [~,index]=sort(Y_TEST,2,'descend');
%   
%   predicted_categories=cell(test_image_number,1);
%   for i=1:test_image_number
%     predicted_categories(i,1)=categories(index(i,1),1);
%   end
% end













