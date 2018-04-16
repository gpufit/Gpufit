#include "com_github_gpufit_Gpufit.h"
#include "Gpufit/gpufit.h"

void * buffer_address(JNIEnv * env, jobject buffer)
{
    if (buffer == 0)
    {
        return 0;
    }
    else
    {
        return env->GetDirectBufferAddress(buffer);
    }
}

/*
* Calls gpufit(), no consistency checks on this side.
*
* Class:     com_github_gpufit_Gpufit
* Method:    fit
* Signature: (IILjava/nio/FloatBuffer;Ljava/nio/FloatBuffer;ILjava/nio/FloatBuffer;FILjava/nio/IntBuffer;IILjava/nio/ByteBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;Ljava/nio/FloatBuffer;Ljava/nio/IntBuffer;)I
*/
jint JNICALL Java_com_github_gpufit_Gpufit_fit(JNIEnv * env, jclass cls, jint number_fits, jint number_points, jobject data_buffer, jobject weights_buffer, jint model_id, jobject initial_parameter_buffer, jfloat tolerance, jint max_number_iterations, jobject paramters_to_fit_buffer, jint estimator_id, jint user_info_size, jobject user_info_buffer, jobject output_parameters_buffer, jobject output_states_buffer, jobject output_chi_squares_buffer, jobject output_number_iterations_buffer)
{
    // get pointer to buffers
    REAL * data = (REAL *)buffer_address(env, data_buffer);
    REAL * weights = (REAL *)buffer_address(env, weights_buffer);
    REAL * initial_parameters = (REAL *)buffer_address(env, initial_parameter_buffer);
    int * parameters_to_fit = (int *)buffer_address(env, paramters_to_fit_buffer);
    char * user_info = (char *)buffer_address(env, user_info_buffer);
    REAL * output_parameters = (REAL *)buffer_address(env, output_parameters_buffer);
    int * output_states = (int *)buffer_address(env, output_states_buffer);
    REAL * output_chi_squares = (REAL *)buffer_address(env, output_chi_squares_buffer);
    int * output_number_iterations = (int *)buffer_address(env, output_number_iterations_buffer);

    // call to gpufit
    int status = gpufit(number_fits, number_points, data, weights, model_id, initial_parameters, tolerance, max_number_iterations, parameters_to_fit, estimator_id, user_info_size, user_info, output_parameters, output_states, output_chi_squares, output_number_iterations);
    return status;
}

/*
* Calls gpufit_get_last_error()
*
* Class:     com_github_gpufit_Gpufit
* Method:    getLastError
* Signature: ()Ljava/lang/String;
*/
jstring JNICALL Java_com_github_gpufit_Gpufit_getLastError(JNIEnv * env, jclass cls)
{
    char const * error = gpufit_get_last_error();
    return env->NewStringUTF(error);
}

/*
* Calls gpufit_cuda_available()
*
* Class:     com_github_gpufit_Gpufit
* Method:    isCudaAvailable
* Signature: ()Z
*/
jboolean JNICALL Java_com_github_gpufit_Gpufit_isCudaAvailable(JNIEnv * env, jclass cls)
{
    return gpufit_cuda_available() == 1 ? JNI_TRUE : JNI_FALSE;
}

/*
* Calls gpufit_get_cuda_version()
*
* Class:     com_github_gpufit_Gpufit
* Method:    getCudaVersionAsArray
* Signature: ()[I
*/
jintArray JNICALL Java_com_github_gpufit_Gpufit_getCudaVersionAsArray(JNIEnv * env, jclass cls)
{
    int runtime_version, driver_version;
    if (gpufit_get_cuda_version(&runtime_version, &driver_version) == ReturnState::OK)
    {
        // create int[2] in Java and fill with values
        jintArray array = env->NewIntArray(2);
        jint fill[2];
        fill[0] = runtime_version;
        fill[1] = driver_version;
        env->SetIntArrayRegion(array, 0, 2, fill);
        return array;
    }
    else
    {
        return 0;
    }
}
