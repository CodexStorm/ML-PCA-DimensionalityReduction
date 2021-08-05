# ML-PCA-DimensionalityReduction

In this project, I will run PCA on face images to see how it can be used in practice for dimension reduction. The dataset `ex7faces.mat` contains a dataset of face images, each  in grayscale. Each row of X corresponds to one face image (a row vector of length 1024). The code in this section will load and visualize the first 100 of these face images.
```
%  Load Face dataset
load ('ex7faces.mat')
%  Display the first 100 faces in the dataset
close all;
displayData(X(1:100, :));
```

## PCA on faces
To run PCA on the face dataset, I first normalize the dataset by subtracting the mean of each feature from the data matrix X. After running PCA, I will obtain the principal components of the dataset. I can visualize these principal components by reshaping each of them into a  matrix that corresponds to the pixels in the original dataset. The code below displays the first 36 principal components that describe the largest variations. Before running PCA, it is important to first normalize X by subtracting the mean value from each feature.
```
[X_norm, ~, ~] = featureNormalize(X);

% Run PCA
[U, ~] = pca(X_norm);

% Visualize the top 36 eigenvectors found
displayData(U(:, 1:36)');
```

The code below will project the face dataset onto only the first 100 principal components. Concretely, each face image is now described by a vector ![](https://latex.codecogs.com/gif.latex?z%5E%7B%28i%29%7D%5Cin%5Cmathbb%7BR%7D%5E%7B100%7D)
```
K = 100;
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: %d x %d', size(Z));
```

To understand what is lost in the dimension reduction, you can recover the data using only the projected dataset. In the code below, an approximate recovery of the data is performed and the original and projected face images are displayed side by side
```
X_rec  = recoverData(Z, U, K);

% Display normalized data
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;
```
