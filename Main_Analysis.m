%% construct the data X1
x = [10*rand(1,30)+5,10*rand(1,30)+5, 10*rand(1,30)+5];
y = [0*rand(1,30),10*rand(1,30)+5, 10*rand(1,30)+5];
z = [10*rand(1,30)+5,0*rand(1,30), 0*rand(1,30)];

X1 = [x;y;z];

%% construct the data X2
x = [10*rand(1,30)+5,10*rand(1,30)+5, 10*rand(1,30)+5];
y = [0*rand(1,30),0*rand(1,30), 10*rand(1,30)+5];
z = [10*rand(1,30)+5,10*rand(1,30)+5, 0*rand(1,30)];

X2 = [x;y;z];

%% get the global discriminative matrix by integrating the data X1 and data X2
% set the parameters for Algorithm 1
lambda1 = 1;
lambda2 = 1;
K1 = 50;
K2 = 50;
% set the parameters for the integrative analysis.
K = min([K1, K2]);%number of neighbors, usually (10~30)
T = 20; %Number of Iterations, usually (10~20)
% the interative process
W = Main_MSCA( X1, K1, lambda1, X2, K2, lambda2, K, T);
% evaluate the sample cluster by ARI
label = [1*ones(1,30),2*ones(1,30),3*ones(1,30)]';
group = SpectralClustering(W,3);
ari = adjrand(label,group)