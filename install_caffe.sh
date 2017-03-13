#!/bin/sh

# Taken from https://gist.github.com/doctorpangloss/f8463bddce2a91b949639522ea1dcbe4, installing caffe is a pain on OSX, I'm using docker, this script might be helpful if you want to install otherwise.

# Install brew
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
# Apple hides old versions of stuff at https://developer.apple.com/download/more/
# Install the latest XCode (8.0).
#   We used to install the XCode Command Line Tools 7.3 here, but that would just upset the most recent versions of brew.
#   So we're going to install all our brew dependencies first, and then downgrade the tools. You can switch back after
#   you have installed caffe.
# Install CUDA toolkit 8.0 release candidate
#   Register and download from https://developer.nvidia.com/cuda-release-candidate-download
#   or this path from https://developer.nvidia.com/compute/cuda/8.0/rc/local_installers/cuda_8.0.29_mac-dmg
#   Select both the driver and the toolkit, no documentation necessary
# Install the experimental NVIDIA Mac drivers
#   Download from http://www.nvidia.com/download/driverResults.aspx/103826/en-us
# Install cuDNN v5 for 8.0 RC or use the latest when it's available
#   Register and download from https://developer.nvidia.com/rdp/cudnn-download 
#   or this path: https://developer.nvidia.com/rdp/assets/cudnn-8.0-osx-x64-v5.0-ga-tgz
#   extract to the NVIDIA CUDA folder and perform necessary linking
#   into your /usr/local/cuda/lib and /usr/local/cuda/include folders
#   You will need to use sudo because the CUDA folder is owned by root
sudo tar -xvf ~/Downloads/cudnn-8.0-osx-x64-v5.0-ga.tar /Developer/NVIDIA/CUDA-8.0/
sudo ln -s /Developer/NVIDIA/CUDA-8.0/lib/libcudnn.dylib /usr/local/cuda/lib/libcudnn.dylib
sudo ln -s /Developer/NVIDIA/CUDA-8.0/lib/libcudnn.5.dylib /usr/local/cuda/lib/libcudnn.5.dylib
sudo ln -s /Developer/NVIDIA/CUDA-8.0/lib/libcudnn_static.a /usr/local/cuda/lib/libcudnn_static.a
sudo ln -s /Developer/NVIDIA/CUDA-8.0/include/cudnn.h /usr/local/cuda/include/cudnn.h
# Install the brew dependencies
#   Do not install python through brew. Only misery lies there
#   We'll use the versions repository to get the right version of boost and boost-python
#   We'll also explicitly upgrade libpng because it's out of date
#   Do not install numpy via brew. Your system python already has it.

brew install -vd snappy leveldb gflags glog szip lmdb
brew install hdf5 opencv
brew upgrade libpng
brew tap homebrew/versions

brew install --build-from-source --with-python -vd protobuf
brew install --build-from-source -vd boost159 boost-python159

# Clone the caffe repo
cd ~/Documents
git clone https://github.com/BVLC/caffe.git
# Setup Makefile.config
#   You can download mine directly from here, but I'll explain all the selections
#     For XCode 7.3:
#       https://www.dropbox.com/s/vuy6ha0p7cc5px3/Makefile.config?dl=1
#     For XCode 8.0 and later (Sierra):
#       https://dl.dropboxusercontent.com/u/2891540/caffe_10.12/Makefile.config
#   First, we'll enable cuDNN
#     USE_CUDNN := 1
#   In order to use the built-in Accelerate.framework, you have to reference it.
#   Astonishingly, nobody has written this anywhere on the internet.
#     BLAS := atlas
#     If you use El Capitan (10.11), we'll use the 10.11 sdk path for vecLib:
#       BLAS_INCLUDE := /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers
#     Otherwise (10.12), let's use the 10.12 sdk path:
#       BLAS_INCLUDE := /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.12.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers
#     BLAS_LIB := /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A
#   Configure to use system python and system numpy
#     PYTHON_INCLUDE := /System/Library/Frameworks/Python.framework/Headers \
#          /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include
#     PYTHON_LIB := /System/Library/Frameworks/Python.framework/Versions/2.7/lib
#   Configure to enable Python layers. Some projects online need this
#     WITH_PYTHON_LAYER := 1
curl https://dl.dropboxusercontent.com/u/2891540/Makefile.config -o Makefile.config
# Download the XCode Command Line Tools for 7.3, since NVIDIA does not yet support Xcode 8.0's tools
#   http://adcdownload.apple.com/Developer_Tools/Command_Line_Tools_OS_X_10.11_for_Xcode_7.3/Command_Line_Tools_OS_X_10.11_for_Xcode_7.3.dmg
# Now, choose those tools instead
sudo xcode-select --switch /Library/Developer/CommandLineTools
# Go ahead and build.
make -j8 all
# To get python going, first we need the dependencies
#   On a super-clean Mac install, you'll need to easy_install pip.
sudo -H easy_install pip
#   Now, we'll install the requirements system-wide. You may also muck about with a virtualenv.
#   Astonishingly, --user is not better known. 
pip install --user -r python/requirements.txt
#   Go ahead and run pytest now. Horrible @rpath warnings which can be ignored.
make -j8 pytest
# Now, install the package
#   Make the distribution folder
make distribute
#   Install the caffe package into your local site-packages
cp -r distribute/python/caffe ~/Library/Python/2.7/lib/python/site-packages/
#   Finally, we have to update references to where the libcaffe libraries are located.
#   You can see how the paths to libraries are referenced relatively
#     otool -L ~/Library/Python/2.7/lib/python/site-packages/caffe/_caffe.so
#   Generally, on a System Integrity Protection -enabled (SIP-enabled) Mac this is no good.
#   So we're just going to change the paths to be direct
cp distribute/lib/libcaffe.so.1.0.0-rc3 ~/Library/Python/2.7/lib/python/site-packages/caffe/libcaffe.so.1.0.0-rc3
install_name_tool -change @rpath/libcaffe.so.1.0.0-rc3 ~/Library/Python/2.7/lib/python/site-packages/caffe/libcaffe.so.1.0.0-rc3 ~/Library/Python/2.7/lib/python/site-packages/caffe/_caffe.so
# Verify that everything works
#   start python and try to import caffe
python -c 'import caffe'
# If you got this far without errors, congratulations, you installed Caffe on a modern Mac OS X 
