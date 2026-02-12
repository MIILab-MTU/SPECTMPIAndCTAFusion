function Zp = apply_clureg_model(Z, model, varargin)
% APPLY_CLUREG_MODEL  Apply closed-form nonrigid registration model to points.
%
%   Zp = apply_clureg_model(Z, model)
%   Zp = apply_clureg_model(Z, model, 'chunk', 2e5, 'return_normalized', false)
%
% Inputs
%   Z     : LxD points to be transformed (double/single). If model.pre exists,
%           Z can be in the original/world coordinates; otherwise Z must be in
%           the SAME normalized coordinate system as model.X.
%   model : struct with fields:
%           - X         : NxD, kernel centers (typically the normalized source)
%           - C         : NxD, kernel coefficients
%           - D         : pxD, polynomial coefficients (optional; empty if none)
%           - mu        : scalar, Laplacian (L1) kernel bandwidth
%           - use_poly  : logical, whether polynomial term is enabled
%           - pre       : (optional) struct with fields:
%                         pre.center (1xD), pre.scale (scalar)
%                         If present, the function will normalize Z using pre
%                         before applying, and denormalize the output.
%
% Name-Value options
%   'chunk'            : positive integer, chunk size for memory-safe K(Z,X)
%                        computation. Default 0 (no chunking).
%   'return_normalized': logical, when model.pre exists:
%                        - false (default): return in WORLD coords (denormalized)
%                        - true : return in NORMALIZED coords
%
% Output
%   Zp : LxD transformed points
%
% Notes
%   - Kernel: Laplacian (cityblock/L1): k(z,x)=exp(-mu*||z-x||_1)
%   - If model.use_poly==true, polynomial basis is [1, z(:,1), ..., z(:,D)]
%   - This is a purely linear apply: Zp = Z + K_{Z,X} * C + P_Z * D
%
% Author: (your name)   Date: (today)

    p = inputParser;
    addParameter(p, 'chunk', 0);
    addParameter(p, 'return_normalized', false);
    parse(p, varargin{:});
    chunk = p.Results.chunk;
    return_normalized = p.Results.return_normalized;

    % ---- Basic checks ----
    validateattributes(Z, {'double','single'}, {'2d','nonempty'});
    Z = double(Z);
    reqFields = {'X','C','mu','use_poly'};
    for f = reqFields
        if ~isfield(model, f{1})
            error('model.%s is required.', f{1});
        end
    end
    X = double(model.X);
    C = double(model.C);
    mu = double(model.mu);
    if ~isfield(model,'D') || isempty(model.D)
        D = zeros(0, size(C,2)); %#ok<ZEROLIKE>
    else
        D = double(model.D);
    end
    use_poly = logical(model.use_poly);

    [L, d] = size(Z);
    if size(X,2) ~= d || size(C,2) ~= d
        error('Dim mismatch: Z(%d), X(%d), C(%d) must have same #cols.', ...
              d, size(X,2), size(C,2));
    end

    % ---- Optional normalize using model.pre ----
    have_pre = isfield(model, 'pre') && isstruct(model.pre) && ...
               isfield(model.pre,'center') && isfield(model.pre,'scale');
    if have_pre
        ctr   = double(model.pre.center(:))';
        scale = double(model.pre.scale);
        if ~isscalar(scale) || ~isfinite(scale) || scale == 0
            error('model.pre.scale must be a finite non-zero scalar.');
        end
        Z_norm = (Z - ctr) ./ scale;
        X_norm = X;   % model.X is already normalized when the model was built
    else
        Z_norm = Z;
        X_norm = X;
    end

    % ---- Build polynomial basis if requested ----
    if use_poly
        Pz_head = @(Zin) [ones(size(Zin,1),1), Zin];  % L x (1+d)
        pcols = 1 + d;
        if ~isempty(D) && size(D,1) ~= pcols
            error('model.D has %d rows, but expected %d (=1+d).', size(D,1), pcols);
        end
    end

    % ---- Compute kernel K_{Z,X} (with optional chunking) ----
    L = size(Z_norm,1);
    N = size(X_norm,1);
    Zp_norm = zeros(L, d);

    compute_update = @(Zblk) Zblk; % identity, we add displacement below

    if chunk > 0 && L*N > chunk
        % Determine row-chunk size ~ chunk/N rows per block
        rows_per = max(1, floor(chunk / max(1,N)));
        for s = 1:rows_per:L
            e = min(L, s + rows_per - 1);
            Zb = Z_norm(s:e, :);
            % Laplacian kernel (cityblock)
            Kzx = exp(-mu * pdist2(Zb, X_norm, 'cityblock'));
            disp_b = Kzx * C;
            if use_poly
                Pz = Pz_head(Zb);
                disp_b = disp_b + Pz * D;
            end
            Zp_norm(s:e, :) = compute_update(Zb) + disp_b;
        end
    else
        Kzx = exp(-mu * pdist2(Z_norm, X_norm, 'cityblock'));  % L x N
        Zp_norm = Z_norm + Kzx * C;
        if use_poly
            Zp_norm = Zp_norm + Pz_head(Z_norm) * D;
        end
    end

    % ---- Denormalize (if model.pre exists and caller wants world coords) ----
    if have_pre && ~return_normalized
        Zp = Zp_norm .* scale + ctr;
    else
        Zp = Zp_norm;
    end
end
