function W = Algorithm_1(X, K, lambda)
n=size(X,2);

window_id_Euclidean = knnsearch(X',X','k',K);
K_neig_matrix_Euclidean = zeros(n,n);
for i = 1:size(window_id_Euclidean)
    K_neig_matrix_Euclidean(i, window_id_Euclidean(i,:)) = 1;
end
K_neig_matrix_1 = K_neig_matrix_Euclidean;
%%
[Z,E] = Algorithm_for_Z(X, lambda, K_neig_matrix_1);
W = (abs(Z)+abs(Z)')./2;
end

