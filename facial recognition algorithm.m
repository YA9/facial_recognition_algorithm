% Exercise 14.1
%{
D = [-1 3; 1 4; 3 4; 7 5; 10 9]
%plot(D(:,1),D(:,2),"x")
D1 = 0.5*[D(:,1) - mean(D(:,1)), D(:,2) - mean(D(:,2))]
hold on
plot(D1(:,1),D1(:,2),"x")
dot = D1'*D1
[V,D] = eig(dot)
V1 = V
quiver(0,0,V1(1,1),V1(2,1))
quiver(0,0,V1(1,2),V1(2,2))
xlim([-5,5])
ylim([-5,5])
reduc = D1*V1(:,2)
B = reduc
plot(-reduc*0.8963,-reduc*0.4435,".")
hold off
%}

% Exercise 14.3
%{
Repo = [b_tr,w_tr,s_tr];
R = transpose(Repo)*Repo;
size(R);
[V,D] = eig(R)
T = [b_new - mean(b_new), w_new - mean(w_new), s_new - mean(s_new)];
A = T*V;
A_new = [A(:,2),A(:,3)];
A_new1 = [0*A(:,1),A(:,2),A(:,3)];
B = A_new1 - A;
hold on
plot3(A(:,1),A(:,2),A(:,3),".")
plot3(A_new1(:,1),A_new1(:,2),A_new1(:,3),"x")
xlim([-50 50])
ylim([-50 50])
zlim([-50 50])
%}

% Exercise 14.5

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

%The following chunk of code doesn't mean anything,
%I was testing out different approaches
%{
%train3 = train2*V;
%ans = train2*smile_train;
%train4 = train2*V1;
%test4 = test2*V1;
%train4-test4;
%var = transpose(train4)*test4;
%train5 = reshape(train4,64,64,10);
%test5 = reshape(test4,64,64,10);
%imagesc(train5(:,:,10));
%imagesc(test5(:,:,10));
%train6 = transpose(train4)*train2;
%test6 = transpose(test4)*test2;
%}


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


