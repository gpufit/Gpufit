#include "../gpufit.h"
#include <array>
#include <iostream>
#include <fstream>
using namespace std;

void dual_gpufit()
{
	// model: y =	a*e^(b*x)+c*e^(d*x);

	std::size_t const n_points{ 8 };
	std::size_t const n_model_parameters{ 4 };

	char timevalue[] = "PATH//TO//samples.dat";
	fstream fileout(timevalue, ios::in | ios::binary);
	streampos begin, end;
	fileout.seekg(0, ios::beg);
	begin = fileout.tellg();
	fileout.seekg(0, ios::end);
	end = fileout.tellg();

	std::size_t filebyteSize = (end - begin);
	cout << "filebyteSize = " << filebyteSize << endl;
	std::size_t n_fits = filebyteSize / 4 / 8;//float type  4 bytes, n_points = 8

	cout << "nfits = " << n_fits << endl;
	//user_info fron data file 
	fileout.seekg(0, ios::beg);
	float* user_info = new float[n_points* n_fits]();
	fileout.read(reinterpret_cast<char*>(user_info), n_fits * n_points * sizeof(float));
	fileout.close();

	int const model_id = DUAL_EXP;

	float *initial_parameters = new float[n_fits * n_model_parameters]();
	for (size_t i = 0; i != n_fits; i++)
	{
		initial_parameters[i * n_model_parameters + 0] = 860.f;
		initial_parameters[i * n_model_parameters + 1] = -0.0448;
		initial_parameters[i * n_model_parameters + 2] = -860.f;
		initial_parameters[i * n_model_parameters + 3] = -0.1429f;
	}

	float * data = new float[n_points * n_fits];
	for (size_t i = 0; i != n_fits; i++)
	{
		data[i*n_points + 0] = 20.f;
		data[i*n_points + 1] = 110.f;
		data[i*n_points + 2] = 180.f;
		data[i*n_points + 3] = 260.f;
		data[i*n_points + 4] = 260.f;
		data[i*n_points + 5] = 180.f;
		data[i*n_points + 6] = 110.f;
		data[i*n_points + 7] = 20.f;
	}


	// tolerance
	float const tolerance = 0.000001f;
	// maximum number of iterations
	int const max_number_iterations = 10;
	// estimator ID
	int const estimator_id = LSE;
	// parameters to fit (all of them)
	std::vector< int > parameters_to_fit(n_model_parameters, 1);

	// output parameters
	float * output_parameters = new float[n_model_parameters * n_fits]();
	int *output_states = new int[n_fits]();
	float * output_chi_square = new float[n_fits]();
	int *output_n_iterations = new int[n_fits]();
	int const gpu_status
		= gpufit
		(
			n_fits,
			n_points,
			data,
			0,
			model_id,
			initial_parameters,
			tolerance,
			max_number_iterations,
			parameters_to_fit.data(),
			estimator_id,
			n_points * n_fits * sizeof(float),
			reinterpret_cast<char *>(user_info),
			output_parameters,
			output_states,
			output_chi_square,
			output_n_iterations
		);

	delete[]initial_parameters;
	delete[]data;
	delete[]user_info;
	delete[]output_chi_square;
	delete[]output_n_iterations;
	delete[]output_states;

	int i = 0;
	while (i<n_fits) {
		cout << "current n point parameters :" << i << endl;
		std::cout << "para.1 = " << output_parameters[0 + i] << std::endl;
		std::cout << "para.2 = " << output_parameters[1 + i] << std::endl;
		std::cout << "para.3 = " << output_parameters[2 + i] << std::endl;
		std::cout << "para.4 = " << output_parameters[3 + i] << std::endl;
		cout << endl;
		//std::cout << "para.1 = " << output_parameters.at(4+i) << std::endl;
		//std::cout << "para.2 = " << output_parameters.at(5+i) << std::endl;
		//std::cout << "para.3 = " << output_parameters.at(6+i) << std::endl;
		//std::cout << "para.4 = " << output_parameters.at(7+i) << std::endl;
		i += n_fits / 5;

	}
	delete[]output_parameters;
}

int main()
{
	dual_gpufit();

	std::cout << std::endl << "Example completed!" << std::endl;
	std::cout << "Press ENTER to exit" << std::endl;
	std::getchar();

	return 0;
}