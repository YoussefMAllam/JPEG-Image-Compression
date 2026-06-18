% --- Parameters ---
img = im2double(imresize(imread('cameraman.tif'), [256 256]));
blockSizes = [4, 8, 16];
retentionRatios = [0.10, 0.25, 0.50];
outputDir = 'results';
if ~exist(outputDir,'dir'), mkdir(outputDir); end

% --- Storage for metrics ---
BlockSize = [];
Retention = [];
Compression = [];
MSE_DCT = [];
MSE_DFT = [];

% --- Main processing loop ---
for B = blockSizes
    fprintf("\n===== Block Size = %dx%d =====\n", B, B);
    for ratio = retentionRatios
        % Process blocks
        [imgDCT, totalC, keptC] = process_blocks(img, B, ratio, 'DCT');
        [imgDFT, ~, ~] = process_blocks(img, B, ratio, 'DFT');

        % Clip to [0,1]
        imgDCT = min(max(imgDCT,0),1);
        imgDFT = min(max(imgDFT,0),1);

        % Compute metrics
        mseDCT = mean((img(:)-imgDCT(:)).^2);
        mseDFT = mean((img(:)-imgDFT(:)).^2);
        compressionRatio = totalC / keptC;

        % Store
        BlockSize(end+1) = B;
        Retention(end+1) = ratio;
        Compression(end+1) = compressionRatio;
        MSE_DCT(end+1) = mseDCT;
        MSE_DFT(end+1) = mseDFT;

        % Display & save images
        fig = figure('Name', sprintf('%dx%d @ %.0f%%', B, B, ratio*100));
        subplot(1,3,1), imshow(img), title('Original');
        subplot(1,3,2), imshow(imgDCT), title(sprintf('DCT (MSE=%.4f)', mseDCT));
        subplot(1,3,3), imshow(imgDFT), title(sprintf('DFT (MSE=%.4f)', mseDFT));
        saveas(fig, sprintf('%s/block_%dx%d_ratio_%02d.png', outputDir, B, B, round(ratio*100)));
        close(fig);
    end
end

% --- Create results table ---
resultsTable = table(BlockSize(:), Retention(:), Compression(:), MSE_DCT(:), MSE_DFT(:), ...
    'VariableNames', {'Block','Retention','CompressionRatio','MSE_DCT','MSE_DFT'});
disp(resultsTable);

% --- Plot side-by-side DCT & DFT MSE ---
plot_mse_comparison(BlockSize, Retention, MSE_DCT, MSE_DFT);

%% === FUNCTION: Process Blocks ===
function [imgRec, totalCoeffs, keptCoeffs] = process_blocks(img, B, ratio, method)
    [rows, cols] = size(img);
    imgRec = zeros(size(img));
    totalCoeffs = 0;
    keptCoeffs = 0;

    for i = 1:B:rows
        for j = 1:B:cols
            block = img(i:i+B-1, j:j+B-1);

            % Transform
            switch method
                case 'DCT'
                    blockTrans = dct2(block);
                case 'DFT'
                    blockTrans = fft2(block);
                otherwise
                    error('Unknown method');
            end

            % Keep top-k coefficients
            vec = blockTrans(:);
            k = round(ratio * numel(vec));
            [~, idx] = sort(abs(vec),'descend');
            vecR = zeros(size(vec));
            vecR(idx(1:k)) = vec(idx(1:k));
            blockR = reshape(vecR, [B B]);

            % Inverse transform
            switch method
                case 'DCT'
                    imgRec(i:i+B-1, j:j+B-1) = idct2(blockR);
                case 'DFT'
                    imgRec(i:i+B-1, j:j+B-1) = real(ifft2(blockR));
            end

            totalCoeffs = totalCoeffs + numel(vec);
            keptCoeffs = keptCoeffs + k;
        end
    end
end

%% === FUNCTION: Plot Side-by-Side DCT & DFT MSE ===
function plot_mse_comparison(BlockSize, Retention, MSE_DCT, MSE_DFT)
    BlockSize   = BlockSize(:);
    Retention   = Retention(:);
    MSE_DCT     = MSE_DCT(:);
    MSE_DFT     = MSE_DFT(:);
    uniqueBlocks = unique(BlockSize);

    figure;
    % --- Left: DCT ---
    subplot(1,2,1); hold on; grid on; box on;
    for B = uniqueBlocks'
        idx = BlockSize==B;
        plot(Retention(idx)*100, MSE_DCT(idx), '-o','LineWidth',1.6);
    end
    title('DCT MSE Across Block Sizes');
    xlabel('Retention (%)'); ylabel('MSE');
    legend(arrayfun(@(x)sprintf('%dx%d',x,x),uniqueBlocks,'UniformOutput',false),'Location','northwest');

    % --- Right: DFT ---
    subplot(1,2,2); hold on; grid on; box on;
    for B = uniqueBlocks'
        idx = BlockSize==B;
        plot(Retention(idx)*100, MSE_DFT(idx), '--s','LineWidth',1.6);
    end
    title('DFT MSE Across Block Sizes');
    xlabel('Retention (%)'); ylabel('MSE');
    legend(arrayfun(@(x)sprintf('%dx%d',x,x),uniqueBlocks,'UniformOutput',false),'Location','northwest');
end
