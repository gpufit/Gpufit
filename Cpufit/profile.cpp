#include "profile.h"
#include <iostream>

profiler_info profiler;

void display_profiler_results()
{
	std::chrono::milliseconds::rep t_all, t_overhead, t;

	t_overhead = t_all = std::chrono::duration_cast<std::chrono::milliseconds>(profiler.all).count();
	std::cout << "\nprofiling results\n";
	std::cout << "total time\t\t" << t_all << " ms (100%) \n";

    t = std::chrono::duration_cast<std::chrono::milliseconds>(profiler.initialize_LM).count();
	t_overhead -= t;
	std::cout << "initialize LM\t\t" << t << " ms (" << (float)t / t_all * 100 << "%) \n";

    t = std::chrono::duration_cast<std::chrono::milliseconds>(profiler.allocate_GPU_memory).count();
    t_overhead -= t;
    std::cout << "allocate GPU memory\t" << t << " ms (" << (float)t / t_all * 100 << "%) \n";

    t = std::chrono::duration_cast<std::chrono::milliseconds>(profiler.copy_data_to_GPU).count();
    t_overhead -= t;
    std::cout << "copy data to GPU\t" << t << " ms (" << (float)t / t_all * 100 << "%) \n";

    t = std::chrono::duration_cast<std::chrono::milliseconds>(profiler.compute_model).count();
    t_overhead -= t;
    std::cout << "compute model\t\t" << t << " ms (" << (float)t / t_all * 100 << "%) \n";

    t = std::chrono::duration_cast<std::chrono::milliseconds>(profiler.compute_chisquare).count();
    t_overhead -= t;
    std::cout << "compute chi-square\t" << t << " ms (" << (float)t / t_all * 100 << "%) \n";

    t = std::chrono::duration_cast<std::chrono::milliseconds>(profiler.compute_gradient).count();
    t_overhead -= t;
    std::cout << "compute gradient\t" << t << " ms (" << (float)t / t_all * 100 << "%) \n";

    t = std::chrono::duration_cast<std::chrono::milliseconds>(profiler.compute_hessian).count();
    t_overhead -= t;
    std::cout << "compute hessian\t\t" << t << " ms (" << (float)t / t_all * 100 << "%) \n";

    t = std::chrono::duration_cast<std::chrono::milliseconds>(profiler.gauss_jordan).count();
    t_overhead -= t;
    std::cout << "Gauss-Jordan\t\t" << t << " ms (" << (float)t / t_all * 100 << "%) \n";

    t = std::chrono::duration_cast<std::chrono::milliseconds>(profiler.evaluate_iteration).count();
    t_overhead -= t;
    std::cout << "evaluate iteration\t" << t << " ms (" << (float)t / t_all * 100 << "%) \n";

    t = std::chrono::duration_cast<std::chrono::milliseconds>(profiler.read_results_from_GPU).count();
    t_overhead -= t;
    std::cout << "read results from GPU\t" << t << " ms (" << (float)t / t_all * 100 << "%) \n";

    t = std::chrono::duration_cast<std::chrono::milliseconds>(profiler.free_GPU_memory).count();
    t_overhead -= t;
    std::cout << "free GPU memory\t\t" << t << " ms (" << (float)t / t_all * 100 << "%) \n";

	std::cout << "remaining time\t\t" << t_overhead << " ms (" << (float)t_overhead / t_all * 100 << "%) \n";	
}
