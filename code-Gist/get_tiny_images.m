% Starter code prepared by James Hays for CS 143, Brown University

%This feature is inspired by the simple tiny images used as features in 
%  80 million tiny images: a large dataset for non-parametric object and
%  scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
%  Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
%  pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/


% image_paths is an N x 1 cell array of strings where each string is an
%  image path on the file system.
% image_feats is an N x d matrix of resized and then vectorized tiny
%  images. E.g. if the images are resized to 16x16, d would equal 256.

% To build a tiny image feature, simply resize the original image to a very
% small square resolution, e.g. 16x16. You can either resize the images to
% square while ignoring their aspect ratio or you can crop the center
% square portion out of each image. Making the tiny images zero mean and
% unit length (normalizing them) will increase performance modestly.

% suggested functions: imread, imresize

function image_feats = get_tiny_images(image_paths)

  number_of_images=size(image_paths,1);
  image_feats=zeros(number_of_images,16*16);
  
  for i=1:number_of_images
      image=imread(image_paths{i});
      
%       高斯滤波基本上没有效果
%       image0=imread(image_paths{i});
%       gau=fspecial('Gaussian',16,8);
%       image=imfilter(image0,gau);

      tiny_image=imresize(image,[16,16]);  % image resize为16*16
      for j=1:16
          for k=1:16
            image_feats(i,(j-1)*16+k)=tiny_image(j,k);  % 将16*16的图像作为特征描述（128维）
          end
      end
      image_feats(i,:)=image_feats(i,:)-mean(image_feats(i,:)); % 每个图像的值减去其平均值，从而实现特征描述的0均值
      image_feats(i,:)=image_feats(i,:)./norm(image_feats(i,:));  % 将特征描述（128维空间的）长度归一化
  end
end


