function dmrg()
    %重整化群
    % 定义哈密顿量参数
    N = 10; % 晶格大小
    t = 1; % 动能项系数
    U = 2; % 库伦相互作用项系数
    J = 0.5; % 交换相互作用项系数
    
    % 构建哈密顿量
    H_0 = zeros(N); % 单电子哈密顿量
    for i=1:N
        H_0(i,i) = -2*t; % 对角线上的项
        if i>1
            H_0(i,i-1) = t; % 对角线下方的项
            H_0(i-1,i) = t; % 对角线上方的项
        end
    end
    
    H_int = zeros(N); % 相互作用项哈密顿量
    for i=1:N
        for j=1:N
            V_ij = (i==j) * U; % 库伦相互作用项
            J_ij = (i~=j) * J; % 交换相互作用项
            H_int(i,j) = V_ij/2 + J_ij;
        end
    end
    
    H = H_0 + H_int; % 总哈密顿量
    
    % 初始密度矩阵
    rho = eye(2^N)/2^N; % 一个纯态系统，因此可以用单位矩阵表示
    
    % DMRG 参数设置
    max_bond_dim = 50; % 最大保留的较大纠缠基数量
    tol = 1e-8; % 截断误差容限
    max_sweeps = 10; % 最大迭代次数
    
    % DMRG 迭代
    for i=1:max_sweeps
        % 对密度矩阵进行重整化
        [rho, trunc_err, energy] = dmrg_step(rho, H, max_bond_dim, tol);
        
        % 输出每次迭代的基态能量和截断误差
        fprintf('Sweep %d, energy = %f, truncation error = %e\n', i, energy, trunc_err);
    end
    
    % 最终基态能量
    fprintf('Ground state energy: %f\n', energy);
end

function [rho_new, trunc_err, energy] = dmrg_step(rho, H, max_bond_dim, tol)
    % DMRG 迭代中的一步
    % rho: 初始密度矩阵
    % H: 哈密顿量
    % max_bond_dim: 最大保留的较大纠缠基数量
    % tol: 截断误差容限
    % rho_new: 更新后的密度矩阵
    % trunc_err: 截断误差
    % energy: 基态能量
    
    % 对密度矩阵进行重整化
    n = size(rho, 1);
    rho = reshape(rho, [2^(n/2), 2^(n/2)]);
    [U, S, V] = svd(rho);
    s = diag(S);
    trunc_err = 1 - sum(s(1:max_bond_dim).^2);
    trunc_err = max(trunc_err, tol);
    s_trunc = s(1:max_bond_dim) / sqrt(sum(s(1:max_bond_dim).^2));
    rho_trunc = U(:,1:max_bond_dim) * diag(s_trunc) * V(:,1:max_bond_dim)';
    
    % 计算基态能量
    H_trunc = kron(H, eye(2^(n/2-1))) + kron(eye(2^(n/2-1)), H_trunc);
    H_trunc = reshape(H_trunc, [2^(n/2), 2^(n/2), 2^(n/2), 2^(n/2)]);
    H_trunc = permute(H_trunc, [1, 3, 2, 4]);
    H_trunc = reshape(H_trunc, [2^n, 2^n]);
    energy = real(trace(rho_trunc * H_trunc));
    
    % 更新密度矩阵
    rho_new = kron(rho_trunc, rho_trunc');
end   