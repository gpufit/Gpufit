package com.github.gpufit.examples;

import com.github.gpufit.*;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Random;

/**
 * Example of the Java binding of the Gpufit library which implements
 * Levenberg Marquardt curve fitting in CUDA
 * https://github.com/gpufit/Gpufit
 *
 * Multiple fits of a 2D Gaussian peak function with Poisson distributed noise
 * http://gpufit.readthedocs.io/en/latest/bindings.html#java
 */
public class Gauss2DExample {

    public static void main(String [] args) {

        // print general CUDA information
        System.out.println(String.format("CUDA available: %b", Gpufit.isCudaAvailable()));
        CudaVersion cudaVersion = Gpufit.getCudaVersion();
        System.out.println(String.format("CUDA versions runtime: %s, driver: %s", cudaVersion.runtime, cudaVersion.driver));

        // number of fits and number of fit points
        int numberFits = 10000;
        int sizeX = 12;
        int numberPoints = sizeX * sizeX;

        // tolerance, maximumNumberIterations, model and estimator
        float tolerance = 0.0001f;
        int maxNumberIterations = 20;
        Model model = Model.GAUSS_2D;
        Estimator estimator = Estimator.MLE;

        // true parameters (order: amplitude, center-x, center-y, width, offset)
        float[] trueParameters = new float[]{10, 5.5f, 5.5f, 3, 10};

        // randomized initial parameters
        Random rand = new Random(0);
        float[] initialParameters = new float[numberFits * model.numberParameters];
        for (int i = 0; i < numberFits; i++) {
            int offset = i * model.numberParameters;
            initialParameters[offset + 0] = trueParameters[0] * (0.8f + 0.4f * rand.nextFloat());
            initialParameters[offset + 1] = trueParameters[1] + trueParameters[3] * (-0.2f + 0.4f * rand.nextFloat());
            initialParameters[offset + 2] = trueParameters[2] + trueParameters[3] * (-0.2f + 0.4f * rand.nextFloat());
            initialParameters[offset + 3] = trueParameters[3] * (0.8f + 0.4f * rand.nextFloat());
            initialParameters[offset + 4] = trueParameters[4] * (0.8f + 0.4f * rand.nextFloat());
        }

        // generate x and y values
        float[] xi = new float[numberPoints];
        float[] yi = new float[numberPoints];
        for (int i = 0; i < sizeX; i++) {
            for (int j = 0; j < sizeX; j++) {
                xi[i * sizeX + j] = i;
                yi[i * sizeX + j] = j;
            }
        }

        // generate data
        float[] gauss2D = generateGauss2D(trueParameters, xi, yi);
        float[] data = new float[numberFits * numberPoints];
        for (int i = 0; i < numberFits; i++) {
            for (int j = 0; j < numberPoints; j++) {
                data[i * numberPoints + j] = gauss2D[j];
            }
        }

        // add Poisson noise
        for (int i = 0; i < numberFits * numberPoints; i++) {
            data[i] = nextPoisson(data[i], rand);
        }

        // assemble FitModel
        FitModel fitModel = new FitModel(numberFits, numberPoints, false, model, tolerance, maxNumberIterations, null, estimator, 0);

        // fill data and initial parameters in the fit model
        fitModel.data.clear();
        fitModel.data.put(data);
        fitModel.initialParameters.clear();
        fitModel.initialParameters.put(initialParameters);

        // fun Gpufit
        FitResults fitResults = Gpufit.fit(fitModel);

        // count FitState outcomes and get a list of those who converged
        boolean[] converged = new boolean[numberFits];
        int numberConverged = 0, numberMaxIterationExceeded = 0, numberSingularHessian = 0, numberNegativeCurvatureMLE = 0;
        for (int i = 0; i < numberFits; i++) {
            FitState fitState = FitState.fromID(fitResults.states.get(i));
            converged[i] = fitState.equals(FitState.CONVERGED);
            switch (fitState) {
                case CONVERGED:
                    numberConverged++;
                    break;
                case MAX_ITERATIONS:
                    numberMaxIterationExceeded++;
                    break;
                case SINGULAR_HESSIAN:
                    numberSingularHessian++;
                    break;
                case NEG_CURVATURE_MLE:
                    numberNegativeCurvatureMLE++;
            }
        }

        // get mean and std of converged parameters
        float [] convergedParameterMean = new float[]{0, 0, 0, 0, 0};
        float [] convergedParameterStd = new float[]{0, 0, 0, 0, 0};
        for (int i = 0; i < numberFits; i++) {
            for (int j = 0; j < model.numberParameters; j++) {
                if (converged[i]) {
                    convergedParameterMean[j] += fitResults.parameters.get(i * model.numberParameters + j);
                }
            }
        }
        for (int i = 0; i < model.numberParameters; i++) {
            convergedParameterMean[i] /= numberConverged;
        }
        for (int i = 0; i < numberFits; i++) {
            for (int j = 0; j < model.numberParameters; j++) {
                if (converged[i]) {
                    float dev = fitResults.parameters.get(i * model.numberParameters + j) - convergedParameterMean[j];
                    convergedParameterStd[j] += dev * dev;
                }
            }
        }
        for (int i = 0; i < model.numberParameters; i++) {
            convergedParameterStd[i] = (float)Math.sqrt(convergedParameterStd[i] / numberConverged);
        }

        // print fit results
        System.out.println("*Gpufit*");
        System.out.println(String.format("Model: %s", model.name()));
        System.out.println(String.format("Number of fits: %d", numberFits));
        System.out.println(String.format("Fit size: %d x %d", sizeX, sizeX));
        System.out.println(String.format("Mean Chi²: %.2f", meanFloatBuffer(fitResults.chiSquares, converged)));
        System.out.println(String.format("Mean  number iterations: %.2f", meanIntBuffer(fitResults.numberIterations, converged)));
        System.out.println(String.format("Time: %.2fs", fitResults.fitDuration));
        System.out.println(String.format("Ratio converged: %.2f %%", (float) numberConverged / numberFits * 100));
        System.out.println(String.format("Ratio max it. exceeded: %.2f %%", (float) numberMaxIterationExceeded / numberFits * 100));
        System.out.println(String.format("Ratio singular Hessian: %.2f %%", (float) numberSingularHessian / numberFits * 100));
        System.out.println(String.format("Ratio neg. curvature MLE: %.2f %%", (float) numberNegativeCurvatureMLE / numberFits * 100));

        System.out.println("\nParameters of 2D Gaussian peak");
        for (int i = 0; i < model.numberParameters; i++) {
            System.out.println(String.format("parameter %d, true: %.2f, mean %.2f, std: %.2f", i, trueParameters[i], convergedParameterMean[i], convergedParameterStd[i]));
        }


    }

    /**
     *
     * @param p
     * @param x
     * @param y
     * @return
     */
    private static float[] generateGauss2D(float[] p, float[] x, float[] y) {
        // checks
        assert(x.length == y.length);
        assert(p.length == 5);

        // calculate data
        float[] data = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            float arg = -((x[i] - p[1]) * (x[i] - p[1]) + (y[i] - p[2]) * (y[i] - p[2])) / (2 * p[3] * p[3]);
            data[i] = p[0] * (float)Math.exp(arg) + p[4];
        }
        return data;
    }

    /**
     * See also: https://en.wikipedia.org/wiki/Inverse_transform_sampling
     * @param lambda
     * @param rand
     * @return
     */
    private static int nextPoisson(float lambda, Random rand) {
        float sum = 0;
        int n = -1;
        while (sum < lambda) {
            n++;
            sum -= Math.log(rand.nextFloat());
        }
        return n;
    }

    /**
     *
     * @param buffer
     * @param mask
     * @return
     */
    private static float meanFloatBuffer(FloatBuffer buffer, boolean[] mask) {
        float sum = 0;
        int n = 0;
        buffer.rewind();
        for (int i = 0; i < buffer.capacity(); i++) {
            float value = buffer.get();
            if (mask[i]) {
                n++;
                sum += value;
            }
        }
        return sum / n;
    }

    /**
     *
     * @param buffer
     * @param mask
     * @return
     */
    private static float meanIntBuffer(IntBuffer buffer, boolean[] mask) {
        float sum = 0;
        int n = 0;
        buffer.rewind();
        for (int i = 0; i < buffer.capacity(); i++) {
            float value = buffer.get();
            if (mask[i]) {
                n++;
                sum += value;
            }
        }
        return (float) sum / n;
    }
}
