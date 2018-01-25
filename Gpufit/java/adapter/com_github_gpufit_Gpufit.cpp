#include "com_github_gpufit_Gpufit.h"
#include "Gpufit/gpufit.h"

/*
 * Class:     com_github_gpufit_Gpufit
 * Method:    add
 * Signature: (II)I
 */
JNIEXPORT jint JNICALL Java_com_github_gpufit_Gpufit_add(JNIEnv * env, jclass cls, jint a, jint b)
{
    return a+b;
}

/*
* Class:     com_github_gpufit_Gpufit
* Method:    fit
* Signature: (IILjava/nio/ByteBuffer;)I
*/
JNIEXPORT jint JNICALL Java_com_github_gpufit_Gpufit_fit(JNIEnv * env, jclass cls, jint number_fits, jint number_points, jobject data_buffer)
{
    float * data = (float *)env->GetDirectBufferAddress(data_buffer);

    for (int i = 0; i < 5; i++)
    {
        data[i] = 3.456f;
    }

    int status = 3;

    return status;
}

/*
* Class:     com_github_gpufit_Gpufit
* Method:    fit
* Signature: (IILjava/nio/FloatBuffer;Ljava/nio/FloatBuffer;Lcom/github/gpufit/Gpufit/ModelID;)I
*/
JNIEXPORT jint JNICALL Java_com_github_gpufit_Gpufit_fit(JNIEnv * env, jclass cls, jint number_fits, jint number_points, jobject x, jobject y, jobject z)
{
    int status;
    // int status = gpufit();
    return status;
}

/*
* Class:     com_github_gpufit_Gpufit
* Method:    getLastError
* Signature: ()Ljava/lang/String;
*/
JNIEXPORT jstring JNICALL Java_com_github_gpufit_Gpufit_getLastError(JNIEnv * env, jclass cls)
{
    char const * error = gpufit_get_last_error();
    return env->NewStringUTF(error);
}

/*
* Class:     com_github_gpufit_Gpufit
* Method:    isCudaAvailable
* Signature: ()Z
*/
JNIEXPORT jboolean JNICALL Java_com_github_gpufit_Gpufit_isCudaAvailable(JNIEnv * env, jclass cls)
{
    return gpufit_cuda_available() == 1 ? JNI_TRUE : JNI_FALSE;
}

/*
* Class:     com_github_gpufit_Gpufit
* Method:    getCudaVersion
* Signature: ()[I
*/
JNIEXPORT jintArray JNICALL Java_com_github_gpufit_Gpufit_getCudaVersion(JNIEnv * env, jclass cls)
{
    int runtime_version, driver_version;
    if (gpufit_get_cuda_version(&runtime_version, &driver_version) == ReturnState::OK)
    {

    }
    else
    {
        return 0;
    }
}