cd cmake-build-debug
cmake -G Ninja .. -DENABLE_CLANG_TIDY=OFF -DCMAKE_INSTALL_PREFIX="../dist" -DCMAKE_BUILD_TYPE:STRING=Debug -DCMAKE_C_COMPILER:FILEPATH="C:\\Program Files\\LLVM\\bin\\clang.exe" -DCMAKE_CXX_COMPILER:FILEPATH="C:\\Program Files\\LLVM\\bin\\clang++.exe" -DCMAKE_RC_COMPILER:FILEPATH="C:\\Program Files\\LLVM\\bin\\llvm-rc.exe"
cmake --build .
cmake --install .
cd ../
