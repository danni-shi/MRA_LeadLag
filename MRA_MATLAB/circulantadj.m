function x = circulantadj(M)
% Adjoint of the linear operator 'circulant'.

    n = size(M, 1);
    assert(size(M, 2) == n);
    
    % If not done already for this size n, build and store a matrix A
    % which represents the operator "adjoint of the circulant function".
    persistent n_save A;
    if isempty(n_save) || n_save ~= n
        n_save = n;
        A = zeros(n, n^2);
        I = eye(n);
        for k = 1 : n
            ek = I(:, k);
            Ck = circulant(ek);
            A(k, :) = Ck(:)';
        end
        A = sparse(A);
    end
    
    x = A*M(:);

end
