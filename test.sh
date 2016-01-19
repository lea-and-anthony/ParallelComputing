cd src_original
echo ------------------------------- ORIGINAL VERSION -------------------------------
make testcpu
cd ../src_modified
echo ------------------------------- MODIFIED VERSION -------------------------------
make testcpu
cd ../src_cuda
echo --------------------------------- CUDA VERSION ---------------------------------
make testgpu
#cd ../src_cuda_optimized
#echo ---------------------------- CUDA OPTIMIZED VERSION ----------------------------
#make testgpu

