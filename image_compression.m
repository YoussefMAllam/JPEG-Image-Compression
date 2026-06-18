% === Load & Resize Image ===
img = imread('cameraman.tif'); 
img = im2double(imresize(img, [256 256]));

blockSizes = [4, 8, 16];
retentionRatios = [0.10, 0.25, 0.50];

outputDir = 'results';
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

% Storage for final comparison tables
BlockSize = [];
Retention = [];
Compression = [];
MSE_DCT = [];
MSE_DFT = [];

for b = 1:length(blockSizes)
    B = blockSizes(b);
    fprintf("\n===== Block Size = %dx%d =====\n", B, B);

    for r = 1:length(retentionRatios)
        ratio = retentionRatios(r);

        imgDCT = zeros(size(img));
        imgDFT = zeros(size(img));

        totalCoeffs = 0;
        keptCoeffs = 0;

        for i = 1:B:size(img,1)
            for j = 1:B:size(img,2)

                block = img(i:i+B-1, j:j+B-1);

                % === DCT ===
                dctBlock = dct2(block);
                vec = dctBlock(:);

                k = round(ratio * length(vec));
                [~, idx] = sort(abs(vec), 'descend');
                keepIdx = idx(1:k);

                vecR = zeros(size(vec));
                vecR(keepIdx) = vec(keepIdx);
                dctBlockR = reshape(vecR, [B B]);

                imgDCT(i:i+B-1, j:j+B-1) = idct2(dctBlockR);

                % === DFT ===
                dftBlock = fft2(block);
                vec2 = dftBlock(:);

                [~, idx2] = sort(abs(vec2), 'descend');
                keepIdx2 = idx2(1:k);

                vecR2 = zeros(size(vec2));
                vecR2(keepIdx2) = vec2(keepIdx2);
                dftBlockR = reshape(vecR2, [B B]);

                imgDFT(i:i+B-1, j:j+B-1) = real(ifft2(dftBlockR));

                % Track compression ratio
                totalCoeffs = totalCoeffs + length(vec);
                keptCoeffs = keptCoeffs + k;
            end
        end

        % Clip
        imgDCT = min(max(imgDCT, 0), 1);
        imgDFT = min(max(imgDFT, 0), 1);

        % Compute metrics
        mseDCT = mean((img(:) - imgDCT(:)).^2);
        mseDFT = mean((img(:) - imgDFT(:)).^2);
        compressionRatio = totalCoeffs / keptCoeffs;

        BlockSize(end+1) = B;
        Retention(end+1) = ratio;
        Compression(end+1) = compressionRatio;
        MSE_DCT(end+1) = mseDCT;
        MSE_DFT(end+1) = mseDFT;

        % Display & save
        fig = figure('Name', sprintf('%dx%d @ %.0f%%', B, B, ratio*100));
        subplot(1,3,1), imshow(img), title('Original');
        subplot(1,3,2), imshow(imgDCT), title(sprintf('DCT (MSE=%.4f)', mseDCT));
        subplot(1,3,3), imshow(imgDFT), title(sprintf('DFT (MSE=%.4f)', mseDFT));

        saveas(fig, sprintf('%s/block_%dx%d_ratio_%02d.png', ...
            outputDir, B, B, round(ratio*100)));
        close(fig);
    end
end

%% === RESULTS TABLE ===
resultsTable = table(BlockSize(:), Retention(:), Compression(:), MSE_DCT(:), MSE_DFT(:), ...
    'VariableNames', {'Block', 'Retention', 'CompressionRatio', 'MSE_DCT', 'MSE_DFT'});

disp(resultsTable);

%% === PLOTS ===
% Normalize vector shapes
BlockSize    = BlockSize(:);
Retention    = Retention(:);
Compression  = Compression(:);
MSE_DCT      = MSE_DCT(:);
MSE_DFT      = MSE_DFT(:);
%% === Ensure all result vectors are column vectors ===
BlockSize    = BlockSize(:);
Retention    = Retention(:);
Compression  = Compression(:);
MSE_DCT      = MSE_DCT(:);
MSE_DFT      = MSE_DFT(:);

%% === Side-by-Side DCT & DFT MSE Plots ===
figure;

% ----- Left plot: DCT -----
subplot(1,2,1); 
hold on; grid on; box on;

for B = unique(BlockSize)'
    idx = BlockSize == B;
    plot(Retention(idx)*100, MSE_DCT(idx), '-o', 'LineWidth', 1.6);
end

title('DCT MSE Across Block Sizes');
xlabel('Retention (%)');
ylabel('MSE');
legend(arrayfun(@(x) sprintf('%dx%d', x, x), unique(BlockSize), 'UniformOutput', false), ...
       'Location','northwest');

% ----- Right plot: DFT -----
subplot(1,2,2); 
hold on; grid on; box on;

for B = unique(BlockSize)'
    idx = BlockSize == B;
    plot(Retention(idx)*100, MSE_DFT(idx), '--s', 'LineWidth', 1.6);
end

title('DFT MSE Across Block Sizes');
xlabel('Retention (%)');
ylabel('MSE');
legend(arrayfun(@(x) sprintf('%dx%d', x, x), unique(BlockSize), 'UniformOutput', false), ...
       'Location','northwest');
