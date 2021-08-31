#!/bin/bash
args=("$@")

# be wary of /i /s /q :: -n? (-s is subfolders?) (-q is no confirmation)

set +v
# create package for Gpufit, assumes everything is compiled

if [[ -z $1 ]]; 
then
    echo specify build base path
    exit 3
fi


if [[ -z $2 ]]; 
then
    echo specify version
    exit 3
fi


if [[ -z $3 ]]; 
then
    echo specify source base path
    exit 3
fi


# define paths

export BUILD_BASE=${args[0]}
export VERSION=${args[1]}
export SOURCE_BASE=${args[2]}

export OUTPUT_NAME=Gpufit_$VERSION
export ROOT_INSTALL=$BUILD_BASE/install
export PACKAGE_INSTALL=$BUILD_BASE/install/$OUTPUT_NAME
export OUTPUT_ZIP=$ROOT_INSTALL/${OUTPUT_NAME}_linux.zip
export OUTPUT_DEB=$ROOT_INSTALL/${OUTPUT_NAME}.deb

export PERFORMANCE_TEST_INSTALL=$PACKAGE_INSTALL/gpufit_performance_test
export PYTHON_INSTALL=$PACKAGE_INSTALL/PYTHON
export x64_MATLAB_INSTALL=$PACKAGE_INSTALL/matlab64
export x64_JAVA_INSTALL=$PACKAGE_INSTALL/java64
export SDK_INSTALL_ROOT=$PACKAGE_INSTALL/gpufit_sdk
export DEB_ROOT=$PACKAGE_INSTALL

export x64_BUILD=$BUILD_BASE
#export x64_BUILD_LIB=$BUILD_BASE/Gpufit


export x64_PYTHON_BUILD=$x64_BUILD/pyGpufit/dist

export x64_MATLAB_BUILD=$x64_BUILD/matlab

export x64_JAVA_BUILD=$x64_BUILD/java

export EXAMPLES_SOURCE=$SOURCE_BASE/examples
export PYTHON_SOURCE=$SOURCE_BASE/Gpufit/python
export MATLAB_SOURCE=$SOURCE_BASE/Gpufit/matlab
export SDK_README_SOURCE=$SOURCE_BASE/package/sdk_readme.txt

export MANUAL_SOURCE=$SOURCE_BASE/docs/_build/latex/Gpufit.pdf
export MANUAL_INSTALL=$PACKAGE_INSTALL/Gpufit_VERSION_Manual.pdf

# clean up (if necessary)

if [[ -e "$ROOT_INSTALL" ]]; 
then 
    rm -r "$ROOT_INSTALL" 
fi
#if [[ -e "$OUTPUT_ZIP" ]]; 
#then 
#    rm "$OUTPUT_ZIP" 
#fi
#if [[ -e "$OUTPUT_DEB" ]]; 
#then 
#    rm "$OUTPUT_DEB" 
#fi

# create root folder

echo create root directory
mkdir $ROOT_INSTALL
mkdir $PACKAGE_INSTALL

# create .deb package
echo collect .deb files
mkdir $DEB_ROOT/DEBIAN
cp "$SOURCE_BASE/package/control" "$DEB_ROOT/DEBIAN"
sed -i "s/version_string/${VERSION}/" $DEB_ROOT/DEBIAN/control

mkdir $DEB_ROOT/usr
mkdir $DEB_ROOT/usr/local
mkdir $DEB_ROOT/usr/lib
cp "$x64_BUILD/Gpufit/libGpufit.so" "$DEB_ROOT/usr/lib"
cp "$x64_BUILD/Cpufit/libCpufit.so" "$DEB_ROOT/usr/lib"

mkdir $DEB_ROOT/usr/local/include
cp "$SOURCE_BASE/Gpufit/gpufit.h" "$DEB_ROOT/usr/local/include"

mkdir $DEB_ROOT/usr/local/bin
cp "$x64_BUILD/Gpufit_Cpufit_Performance_Comparison" "$DEB_ROOT/usr/local/bin"

echo creating .deb
(cd ${DEB_ROOT}/.. && dpkg-deb --build $OUTPUT_NAME)

echo delete temp files
rm -r $DEB_ROOT
mkdir $PACKAGE_INSTALL
mv ${OUTPUT_DEB} ${PACKAGE_INSTALL}

# copy main readme (is markdown, written as txt) and license.

cp "$SOURCE_BASE/README.md" "$PACKAGE_INSTALL/README.txt"
cp "$SOURCE_BASE/LICENSE.txt" "$PACKAGE_INSTALL"

# copy manual

#if [[ ! -e "$MANUAL_SOURCE" ]]; 
#then
#    echo file $MANUAL_SOURCE required, does not exist
#    exit 3
#fi
#
#cp "$MANUAL_SOURCE" "$MANUAL_INSTALL"

# copy performance test

echo collect performance test application
mkdir $PERFORMANCE_TEST_INSTALL
cp "$EXAMPLES_SOURCE/Gpufit_Cpufit_Performance_Comparison_readme.txt" "$PERFORMANCE_TEST_INSTALL/README.txt"

mkdir $PERFORMANCE_TEST_INSTALL/linux
cp "$x64_BUILD/Gpufit_Cpufit_Performance_Comparison" "$PERFORMANCE_TEST_INSTALL/linux"
cp "$x64_BUILD/Gpufit/libGpufit.so" "$PERFORMANCE_TEST_INSTALL/linux"
cp "$x64_BUILD/Cpufit/libCpufit.so" "$PERFORMANCE_TEST_INSTALL/linux"


# copy Python packages

echo collect python
mkdir $PYTHON_INSTALL
cp "$x64_PYTHON_BUILD/pyGpufit-$VERSION-py2.py3-none-any.whl" "$PYTHON_INSTALL/pyGpufit-$VERSION-py2.py3-none-any.whl"
cp "$PYTHON_SOURCE/README.txt" "$PYTHON_INSTALL"
cp "$PYTHON_SOURCE/examples" "$PYTHON_INSTALL/examples" -n -r


# copy Matlab 64 bit

echo collect matlab64
mkdir $x64_MATLAB_INSTALL
cp "$x64_MATLAB_BUILD" "$x64_MATLAB_INSTALL" -r
cp "$MATLAB_SOURCE/examples" "$x64_MATLAB_INSTALL/examples" -n -r


# copy Java 64 bit

echo collect java64
mkdir $x64_JAVA_INSTALL
cp "$x64_JAVA_BUILD" "$x64_JAVA_INSTALL" -r


# copy SDK_INSTALL_ROOT

echo collect SDK
mkdir $SDK_INSTALL_ROOT
cp "$SDK_README_SOURCE" "$SDK_INSTALL_ROOT/README.txt"

mkdir $SDK_INSTALL_ROOT/include
cp "$SOURCE_BASE/Gpufit/gpufit.h" "$SDK_INSTALL_ROOT/include"

mkdir $SDK_INSTALL_ROOT/linux
cp "$x64_BUILD/Gpufit/libGpufit.so" "$SDK_INSTALL_ROOT/linux"
#cp "$x64_BUILD_LIB/Gpufit.a" "$SDK_INSTALL_ROOT/linux"


# zip content of temp folder with 7-Zip if available
if 7z a -bso0 "${OUTPUT_ZIP}" "${PACKAGE_INSTALL}"
then
    #it worked
    echo Created ${OUTPUT_ZIP}
else
    echo Packing ${OUTPUT_ZIP} failed
fi


