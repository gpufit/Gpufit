#!/bin/bash

printf "Gauss 2D Rotated\n"
./Gpufit_Test_Gauss_Fit_2D_Rotated

printf "\nGauss 2D Elliptic\n"
./Gpufit_Test_Gauss_Fit_2D_Elliptic

printf "\nGauss 2D\n"
./Gpufit_Test_Gauss_Fit_2D

printf "\nGauss 1D\n"
./Gpufit_Test_Gauss_Fit_1D

printf "\nLinear 1D\n"
./Gpufit_Test_Linear_Fit_1D

printf "\nFletcher Powell Helix\n"
./Gpufit_Test_Fletcher_Powell_Helix_Fit

printf "\nError Handling\n"
./Gpufit_Test_Error_Handling

printf "\nCauchy Fit 2D Elliptic\n"
./Gpufit_Test_Cauchy_Fit_2D_Elliptic

printf "\nBrown Dennis\n"
./Gpufit_Test_Brown_Dennis_Fit

printf "\nTest Consistency\n"
./Cpufit_Gpufit_Test_Consistency



