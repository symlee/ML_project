A = single(rand(2000));
A2 = gpuArray(A);
gd =gpuDevice;
tic;b1 = fft(A); toc
tic;b1 = fft(A2); wait(gd); toc
tic;b1 = A*A; toc
tic;b1 = A2*A2; wait(gd); toc

