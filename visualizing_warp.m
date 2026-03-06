%% XOR (2D) -> Learned Warp to 3D (tanh) with 2-panel visualization + AVI export
clear; close all; clc;

%% 1) Create XOR data in 2D
rng(0);
N = 200;
noise = 0.25;

x = sign(randn(N,1));
y = sign(randn(N,1));

X = [x, y] + noise*randn(N,2);
t = double((x .* y) < 0);

% --- fixed class colors
c0 = [0 0.447 0.741];   % blue
c1 = [0.85 0.325 0.098]; % red/orange

%% 2) Build input grid
pad = 0.8;
xmin = min(X(:,1)) - pad; xmax = max(X(:,1)) + pad;
ymin = min(X(:,2)) - pad; ymax = max(X(:,2)) + pad;

m = 30;
[gx, gy] = meshgrid(linspace(xmin,xmax,m), linspace(ymin,ymax,m));
G = [gx(:), gy(:)];

%% 3) Network
rng(1);

H = 16;
lr = 0.05;
T  = 200;
snapEvery = 2;
pauseSec = 0.12;

phi  = @(u) tanh(u);
dphi = @(u) 1 - tanh(u).^2;
sig  = @(u) 1 ./ (1 + exp(-u));

W1 = 0.6*randn(2,H);   b1 = zeros(1,H);
W2 = 0.6*randn(H,3);   b2 = zeros(1,3);
W3 = 0.6*randn(3,1);   b3 = 0;

%% 4) Setup figure
fig = figure('Color','w','Position',[100 100 1400 550]);

% Left: original XOR
subplot(1,2,1); hold on;
scatter(X(t==0,1), X(t==0,2), 40, c0, 'filled');
scatter(X(t==1,1), X(t==1,2), 40, c1, 'filled');
axis equal; grid on;
title('Original XOR in 2D');
xlabel('x'); ylabel('y');
xlim([xmin xmax]); ylim([ymin ymax]);

az = 30; el = 15;

fixedAxes = false;
lims3 = struct('x',[],'y',[],'z',[]);

vidName = 'xor_warp_training.avi';
v = VideoWriter(vidName, 'Motion JPEG AVI');
v.FrameRate = 10;
v.Quality = 95;
open(v);

%% 6) Train + visualize
for step = 1:T

    % Forward
    z1 = X*W1 + b1;     h1 = phi(z1);
    z2 = h1*W2 + b2;    h2 = phi(z2);
    z3 = h2*W3 + b3;    p  = sig(z3);

    % Loss
    eps_ = 1e-9;
    loss = -mean(t.*log(p+eps_) + (1-t).*log(1-p+eps_));

    % Backprop
    Ncur = size(X,1);
    dz3 = (p - t) / Ncur;

    dW3 = h2' * dz3;
    db3 = sum(dz3,1);

    dh2 = dz3 * W3';
    dz2 = dh2 .* dphi(z2);

    dW2 = h1' * dz2;
    db2 = sum(dz2,1);

    dh1 = dz2 * W2';
    dz1 = dh1 .* dphi(z1);

    dW1 = X' * dz1;
    db1 = sum(dz1,1);

    % Update
    W1 = W1 - lr*dW1;  b1 = b1 - lr*db1;
    W2 = W2 - lr*dW2;  b2 = b2 - lr*db2;
    W3 = W3 - lr*dW3;  b3 = b3 - lr*db3;

    % Visualize
    if step == 1 || mod(step, snapEvery) == 0

        acc = mean((p >= 0.5) == t);

        z1g = G*W1 + b1;     h1g = phi(z1g);
        z2g = h1g*W2 + b2;   h2g = phi(z2g);
        Hg = reshape(h2g, [m, m, 3]);

        subplot(1,2,2); cla; hold on; grid on;

        % Wireframe
        for r = 1:m
            plot3(Hg(r,:,1), Hg(r,:,2), Hg(r,:,3), 'k-', 'LineWidth', 0.6);
        end
        for c = 1:m
            plot3(Hg(:,c,1), Hg(:,c,2), Hg(:,c,3), 'k-', 'LineWidth', 0.6);
        end

        % Colored points (MATCH LEFT PLOT)
        scatter3(h2(t==0,1), h2(t==0,2), h2(t==0,3), 40, c0, 'filled');
        scatter3(h2(t==1,1), h2(t==1,2), h2(t==1,3), 40, c1, 'filled');

        % Separating plane
        w = W3(:);
        if abs(w(3)) > 1e-6
            xl = xlim; yl = ylim;
            [px, py] = meshgrid(linspace(xl(1),xl(2),10), ...
                                linspace(yl(1),yl(2),10));
            pz = -(w(1)*px + w(2)*py + b3) / w(3);
            surf(px, py, pz, ...
                'FaceAlpha', 0.35, ...
                'EdgeColor', 'none');
             colormap(parula);     % smooth gradient
             shading interp;       % smooth color blending

        end

        xlabel('h1'); ylabel('h2'); zlabel('h3');
        view(az, el);
        axis vis3d;

        title(sprintf('3D warp (%d°, %d°) | step %d | loss %.3f | acc %.2f', ...
            az, el, step, loss, acc));

        if ~fixedAxes
            lims3.x = xlim; lims3.y = ylim; lims3.z = zlim;
            fixedAxes = true;
        else
            xlim(lims3.x); ylim(lims3.y); zlim(lims3.z);
        end

        drawnow;

        frame = getframe(fig);
        writeVideo(v, frame);

        pause(pauseSec);
    end
end

close(v);
aviPath = fullfile(pwd, 'xor_warp_training.avi');
mp4Path = fullfile(pwd, 'xor_warp_training.mp4');

cmd = sprintf('ffmpeg -i "%s" -vcodec libx264 -pix_fmt yuv420p "%s"', aviPath, mp4Path);
system(cmd);
fprintf('Saved video: %s\n', vidName);