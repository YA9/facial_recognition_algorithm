%saving the training data into the variables 'train' and 'test'
train = grayfaces_train;
test = grayfaces_test;

%reshaping the train and test data so that there is a full image per column
train1 = reshape(train,4096,356);
test1 = reshape(test,4096,356);

%mean-centering the data by subtracting the mean of train1 from the train
%and test data
train2 = train1 - mean(train1);
test2 = test1 - mean(train1);

%creating the correlation matrix (356x356)
trained = transpose(train1)*train1;

%getting the eigenvectors and values for the correlation matrix
[V,D] = eig(trained);
d = nonzeros(D);

%The first for-loop iterates through different numbers of eigenfaces with
%the highest eigenvalues to see their accuracies.
for i = 1:100
    %This is where the iteration is happening for the selection of
    %eigenvectors
    D1 = D(:,(357-i):356);
    V1 = V(:,(357-i):356);
    
    %I create the 'i' number of eigenfaces for the algorithm
    train4 = train2*V1;
    test4 = test2*V1;
    
    %Getting the original data in terms of eigenfaces (converting to
    %face-space)
    train6 = transpose(train4)*train2;
    test6 = transpose(test4)*test2;
    
    %iterating through all the faces to find the closest match (the actual
    %facial recognition algorithm)
    test_face = 0*test6(:,1);
    for p = 1:356;
        
        %Calculating the distances between the selected face and every
        %other face
        for x = 1:356;
        dists = (test6(:,p)-train6(:,x)).^2;
        dist = sum(dists);
        test_face(x) = dist;
        end
    test_face1 = sqrt(test_face);
    
    %Finding the matched face (I)
    [M,I] = min(test_face1);
    
    %Calculating the accuracy for the certain number of eigenfaces
    acc(p) = subject_train(p,:)==subject_test(I,:);
    end
    final(i) = mean(acc);
end

%Plotting the 'Accuracy' vs 'Number of principal components' graph
plot(1:292,final)
xlabel('Number of principal components');
ylabel('Accuracy');

%The following visually compares the matched face to the original face
%imagesc(train(:,:,p))
%figure
%imagesc(test(:,:,I))


