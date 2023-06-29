function [x_aligned, E, perm] = align_to_reference_het(x, xref)
% Input: x and xref both of size NxK
% Output: x_aligned of size NxK: contains the same data as x, with columns
% permuted and circularly shifted (individually) to match xref as closely
% as possible (in some sense defined by the code.)

    assert(all(size(x) == size(xref)), 'x and xref must have identical size');
    
    K = size(x, 2);
    

    E = zeros(K);
    for k1 = 1 : K
        for k2 = 1 : K
            E(k1, k2) = norm(align_to_reference(x(:, k2), xref(:, k1)) - xref(:, k1), 2)^2;
        end
    end

    % Out-sourced implementation of Hungarian algorithm.
    perm = munkres(E);
    
    x_aligned = x(:, perm);
    
    for k = 1 : K
        x_aligned(:, k) = align_to_reference(x_aligned(:, k), xref(:, k));
    end

end




% function [indices, x_aligned, E, perm] = align_to_reference_het(x, xref)
% % Input: x and xref both of size NxK
% % Output: x_aligned of size NxK: contains the same data as x, with columns
% % permuted and circularly shifted (individually) to match xref as closely
% % as possible (in some sense defined by the code.)
% 
%     assert(all(size(x) == size(xref)), 'x and xref must have identical size');
%     
%     K = size(x, 2);
%     
% 
%     E = zeros(K);
%     indices = zeros(K,1);
%     for k1 = 1 : K
%         for k2 = 1 : K
%             E(k1, k2) = norm(align_to_reference(x(:, k2), xref(:, k1)) - xref(:, k1), 2)^2;
%         end
%     end
% 
%     % Out-sourced implementation of Hungarian algorithm.
%     perm = munkres(E);
%     
%     x_aligned = x(:, perm);
%     
%     for k = 1 : K
%         [ind, x_aligned(:, k)] = align_to_reference(x_aligned(:, k), xref(:, k));
%         indices(k) = ind;
%     end
% 
% end
