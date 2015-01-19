function [patches, labels]= loadData(patchsize,numpatches,is_test)

%add MNIST data path
addpath('..//MNIST');

%load train / test images
if is_test == 0
	IMAGES = loadMNISTImages('..//MNIST//train-images');

else
	IMAGES = loadMNISTImages('..//MNIST//test-images');
end

n = sqrt(size(IMAGES,1));
IMAGES = reshape(IMAGES,[n n size(IMAGES,2)]);
LABELS = getLabels(is_test);

patches = zeros(patchsize*patchsize, numpatches);
labels = zeros(10,numpatches);

rng('shuffle');
for i = 1:numpatches
	%randomly select an image
	[im_h im_w no_images] = size(IMAGES);

	%take full training set
	image_no = i;			

	%in case we want to subsample from the image set
	%image_no = round(rand(1) * (no_images-1)) + 1;

	%randomly select left corner of patch
	lc_x = round(rand(1) * (im_w - 1)) + 1;
	lc_y = round(rand(1) * (im_h - 1)) + 1;

	%check if the patch lies within the image boundaries, if not pull the patch inside the boundary
	if lc_x + patchsize-1 > im_w
		lc_x = lc_x + (im_w - (lc_x + patchsize)) + 1;
	end
	if lc_y + patchsize-1 > im_h
		lc_y = lc_y + (im_h - (lc_y + patchsize)) + 1;
	end

	patch = IMAGES(lc_x:lc_x+patchsize-1, lc_y:lc_y+patchsize-1,image_no);
	patches(:,i) = patch(:);
	labels(:,i) = LABELS(:,image_no);
end

patches = normalizeData(patches);

end


