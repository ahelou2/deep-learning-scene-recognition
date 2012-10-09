function xyz = image2WorldCoords(uvr,image_size,R)

% Transform from image to world coordinates.

f = 1;
u0 = image_size(1)/2;
v0 = image_size(2)/2;
num_coords = size(uvr,1);
uvr = double(uvr) - repmat([u0, v0, 0],num_coords,1);
uvr = uvr/v0;
xyz = [uvr(:,1), uvr(:,2), f*ones(num_coords,1)];
relative_size = R./uvr(:,3);
xyz = xyz.*repmat(relative_size, 1, 3);

% Take the log of z since log(z) is Gaussian
xyz(:,3) = log(xyz(:,3));

