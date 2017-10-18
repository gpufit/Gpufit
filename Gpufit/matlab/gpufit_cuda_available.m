function r = gpufit_cuda_available()
% Returns true if CUDA is available and false otherwise

r = GpufitCudaAvailableMex() == 1;

end