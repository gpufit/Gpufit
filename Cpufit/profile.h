#ifndef CPUFIT_PROFILE_H_INCLUDED
#define CPUFIT_PROFILE_H_INCLUDED

#include <chrono>

struct profiler_info
{
	std::chrono::high_resolution_clock::duration initialize_LM, all, allocate_GPU_memory, copy_data_to_GPU, read_results_from_GPU, compute_model, compute_chisquare, compute_gradient, compute_hessian, gauss_jordan, evaluate_iteration, free_GPU_memory;
};

extern profiler_info profiler;

void display_profiler_results();

#endif