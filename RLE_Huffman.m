%% === Load & Resize Image ===
img = im2double(imresize(imread('cameraman.tif'), [256 256]));

% Explicit normalization to [0,1]
img = (img - min(img(:))) / (max(img(:)) - min(img(:)));
B = 8;                    
ratio = 0.25;             
imgDCT = zeros(size(img));

B = 8;                    
ratio = 0.25;             
imgDCT = zeros(size(img));

% Precompute zigzag indices
zigzagIdx = zigzag_indices(B);

% Storage for encoded data
encodedBlocks = cell(size(img)/B);

%% === Process Each Block (Compression) ===
for i = 1:B:size(img,1)
    for j = 1:B:size(img,2)
        block = img(i:i+B-1, j:j+B-1);
        dctBlock = dct2(block);
        
        % --- Keep top-k coefficients ---
        vec = dctBlock(:);
        k = round(ratio * numel(vec));
        [~, idx] = sort(abs(vec), 'descend');
        keepIdx = idx(1:k);
        reduced = zeros(size(vec));
        reduced(keepIdx) = vec(keepIdx);
        dctBlockR = reshape(reduced, [B B]);

        % --- Zigzag Scan ---
        zz = dctBlockR(zigzagIdx);

        % --- Run-Length Encoding ---
        rle = run_length_encode(zz);
        encodedBlocks{(i-1)/B+1, (j-1)/B+1} = rle;
    end
end

%% === Combine All RLE Streams ===
allRLE = [];
for a = 1:numel(encodedBlocks)
    blockRLE = encodedBlocks{a};
    allRLE = [allRLE; blockRLE]; 
end

% Convert RLE pairs into string symbols for Huffman coding
symbols = cell(size(allRLE,1),1);
for i = 1:length(symbols)
    symbols{i} = sprintf('%g,%g', allRLE(i,1), allRLE(i,2));
end

%% === Build Huffman Dictionary ===
[uniqueSymbols,~,idx] = unique(symbols);
counts = histcounts(idx, 0.5:1:(max(idx)+0.5));
prob = counts / sum(counts);
dict = huffmandict(uniqueSymbols, prob);

%% === Huffman Encoding ===
encodedStream = huffmanenco(symbols, dict);

%% === Compression Statistics ===
original_bits = numel(img) * 8; % assume 8 bits per number
encoded_bits = length(encodedStream);
compression_ratio = original_bits / encoded_bits;

fprintf('Estimated Compression Ratio = %.2f:1\n', compression_ratio);

% === Average Bits Per Pixel ===
num_pixels = numel(img);
bpp = encoded_bits / num_pixels;
fprintf('Average Bits Per Pixel = %.4f bits/pixel\n', bpp);

%% === Huffman Decoding ===
decodedSymbols = huffmandeco(encodedStream, dict);
assert(isequal(symbols, decodedSymbols), 'Huffman decoding mismatch');
disp('Huffman encoding and decoding successful');

%% === Reconstruct All Blocks ===
decodedRLE = zeros(size(allRLE));
for i = 1:length(decodedSymbols)
    vals = sscanf(decodedSymbols{i}, '%f,%f');
    decodedRLE(i,:) = vals';
end

% Split back into cell array per block
decodedBlocks = cell(size(encodedBlocks));
count = 1;
for a = 1:numel(decodedBlocks)
    rle = [];
    while count <= size(decodedRLE,1)
        rle = [rle; decodedRLE(count,:)];
        count = count + 1;
        % heuristic break — each block ends when enough coefficients (B*B)
        if sum(rle(:,2)==0) >= ratio*(B*B)
            break;
        end
    end
    decodedBlocks{a} = rle;
end

%% === RLE Decoding + Inverse DCT ===
imgRec = zeros(size(img));
blockIdx = 1;
for i = 1:B:size(img,1)
    for j = 1:B:size(img,2)
        rle = encodedBlocks{(i-1)/B+1, (j-1)/B+1};
        zz = run_length_decode(rle, B*B);          
        blockDCT = zeros(B*B,1);
        blockDCT(zigzagIdx) = zz;                 % inverse zigzag
        blockDCT = reshape(blockDCT, [B B]);
        block = idct2(blockDCT);                  % inverse DCT
        imgRec(i:i+B-1, j:j+B-1) = block;
        blockIdx = blockIdx + 1;
    end
end

imgRec = (imgRec - min(imgRec(:))) / (max(imgRec(:)) - min(imgRec(:)));
%% === Display Original vs Reconstructed ===
figure;
subplot(1,2,1);
imshow(img, []);
title('Original Image');

subplot(1,2,2);
imshow(imgRec, []);
title(sprintf('Reconstructed Image (ratio=%.2f, bpp=%.2f)', ratio, bpp));
%% === Helper Functions ===
function idx = zigzag_indices(n)
    % Generate zigzag scan order indices for an n×n block
    idx = zeros(n,n);
    count = 1;
    for s = 1:2*n-1
        if mod(s,2)==0
            r = max(1, s-n+1):min(n,s);
        else
            r = min(n,s):-1:max(1, s-n+1);
        end
        c = s - r + 1;
        for k = 1:length(r)
            idx(r(k), c(k)) = count;
            count = count + 1;
        end
    end
    [~, order] = sort(idx(:));
    idx = order;
end

function rle = run_length_encode(seq)
    rle = [];
    count = 0;
    for i = 1:length(seq)
        if seq(i) == 0
            count = count + 1;
        else
            if count > 0
                rle = [rle; 0 count];
                count = 0;
            end
            rle = [rle; seq(i) 0];
        end
    end
    if count > 0
        rle = [rle; 0 count];
    end
end

function seq = run_length_decode(rle, N)
    seq = [];
    for i = 1:size(rle,1)
        val = rle(i,1);
        run = rle(i,2);
        if val == 0
            seq = [seq; zeros(run,1)]; 
        else
            seq = [seq; val];
        end
    end
    if length(seq) < N
        seq = [seq; zeros(N - length(seq), 1)];
    else
        seq = seq(1:N);
    end
end

