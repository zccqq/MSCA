function W = Main_MSCA( X1, K1, lambda1, X2, K2, lambda2, K, T)
% get local sample discriminative matrix of data X1
W1 = Algorithm_1(X1, K1, lambda1);
% get local sample discriminative matrix of data X2
W2 = Algorithm_1(X2, K2, lambda2);
% get global sample discriminative matrix by integrating the data X1 and X2
W=Algorithm_for_integration({W1,W2},K,T);
end

