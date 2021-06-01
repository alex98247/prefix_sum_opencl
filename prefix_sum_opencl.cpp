#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <chrono>
#include <fstream>
#include <CL/cl.h>
#include <chrono>
#include <vector>

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::steady_clock;

#define MAX_SOURCE_SIZE (0x100000)

int main(void) {
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    std::vector <cl_device_id> cpu_devices;
    std::vector <cl_device_id> gpu_descrete_devices;
    std::vector <cl_device_id> gpu_nondescrete_devices;
    cl_ulong device_host_unified_memory;
    cl_device_type device_type;
    cl_int ret;

    if (clGetPlatformIDs(0, NULL, &platformCount) != CL_SUCCESS) {
        std::cerr << "Read platforms count error" << std::endl;
        return 1;
    }
    platforms = new cl_platform_id[platformCount];
    if (!platforms) {
        std::cerr << "Can not allocate memory" << std::endl;
        delete[] platforms;
        return 1;
    }
    if (clGetPlatformIDs(platformCount, platforms, NULL) != CL_SUCCESS) {
        std::cerr << "Read platforms error" << std::endl;
        delete [] platforms;
        return 1;
    }

    for (int i = 0; i < platformCount; i++) {

        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount) != CL_SUCCESS) {
            std::cerr << "Read devices count error" << std::endl;
            delete[] platforms;
            return 1;
        }
        devices = new cl_device_id[deviceCount];
        if (!devices) {
            std::cerr << "Can not allocate memory" << std::endl;
            delete[] platforms;
            delete[] devices;
            return 1;
        }
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL) != CL_SUCCESS) {
            std::cerr << "Read devices error" << std::endl;
            delete[] platforms;
            delete[] devices;
            return 1;
        }

        for (int j = 0; j < deviceCount; j++) {
            if (clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL) != CL_SUCCESS) {
                std::cerr << "Get device info error" << std::endl;
                delete[] platforms;
                delete[] devices;
                return 1;
            }
            if (device_type == CL_DEVICE_TYPE_CPU) {
                cpu_devices.push_back(devices[j]);
            }
            if (device_type == CL_DEVICE_TYPE_GPU) {
                if(clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_ulong), &device_host_unified_memory, NULL) != CL_SUCCESS) {
                    std::cerr << "Get device info error" << std::endl;
                    delete[] platforms;
                    delete[] devices;
                    return 1;
                }
                if (device_host_unified_memory == CL_FALSE) {
                    gpu_descrete_devices.push_back(devices[j]);
                }
                else {
                    gpu_nondescrete_devices.push_back(devices[j]);
                }
            }

        }

        delete[] devices;
    }

    delete[] platforms;

    std::string file_path = "C:\\Users\\Alex\\source\\repos\\ConsoleApplication1\\test1.txt";//argv[1];
    int n = 0;
    float* A;
    std::ifstream in(file_path);

    if (in.is_open())
    {
        in >> n;
        if (in.fail()) {
            std::cerr << "Wrong input value" << std::endl;
            return 1;
        }

        A = new float[n];
        if (!A) {
            std::cerr << "Can not allocate memory" << std::endl;
            delete[] A;
            return 1;
        }

        for (int i = 0; i < n; i++)
        {
            in >> A[i];
            if (in.fail()) {
                std::cerr << "Wrong input value" << std::endl;
                delete[] A;
                return 1;
            }
        }
    }
    else
    {
        std::cerr << "Can not read file " << file_path << std::endl;
        return 1;
    }
    in.close();

    FILE* fp;
    char* source_str;
    size_t source_size;

    fopen_s(&fp, "C:\\Users\\Alex\\source\\repos\\ConsoleApplication1\\2.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        return 1;
    }
    source_str = new char [MAX_SOURCE_SIZE];
    if (!source_str) {
        std::cerr << "Can not allocate memory" << std::endl;
        delete[] A;
        delete[] source_str;
        return 1;
    }
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);


    int global_id_device = 1;
    cl_device_id* device_id = NULL;
    if (global_id_device < 0) {
        return 1;
    }
    if (global_id_device < gpu_descrete_devices.size()) {
        device_id = &gpu_descrete_devices[global_id_device];
    }
    else if (global_id_device < gpu_descrete_devices.size() + gpu_nondescrete_devices.size()) {
        device_id = &gpu_nondescrete_devices[global_id_device - gpu_descrete_devices.size()];
    }
    else if (global_id_device < gpu_descrete_devices.size() + gpu_nondescrete_devices.size() + cpu_devices.size()) {
        device_id = &cpu_devices[global_id_device - gpu_nondescrete_devices.size() - gpu_descrete_devices.size()];
    }
    else {
        return 1;
    }

    size_t global_item_size = n;
    size_t local_item_size = 64;
    if (global_item_size % local_item_size != 0) {
        global_item_size = global_item_size - global_item_size % local_item_size + local_item_size;
    }

    size_t global_groups_size = (global_item_size / local_item_size);
    if (global_groups_size % local_item_size != 0) {
        global_groups_size = global_groups_size - global_groups_size % local_item_size + local_item_size;
    }

    steady_clock::time_point start = high_resolution_clock::now();
    cl_context context = clCreateContext(NULL, 1, device_id, NULL, NULL, &ret);

    cl_command_queue command_queue = clCreateCommandQueue(context, *device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float), NULL, &ret);
    cl_mem max_sum_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, global_groups_size * sizeof(float), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, n * sizeof(float), A, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);

    ret = clBuildProgram(program, 1, device_id, NULL, NULL, NULL);

    if (ret == CL_BUILD_PROGRAM_FAILURE
#ifdef DEBUG
        || errCode == 0
#endif
        ) {
        size_t clBuildInfoLogSize = 0;
        clGetProgramBuildInfo(program, *device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &clBuildInfoLogSize);
        char* buildInfoLog = new char[clBuildInfoLogSize];
        clGetProgramBuildInfo(program, *device_id, CL_PROGRAM_BUILD_LOG, clBuildInfoLogSize, buildInfoLog, &clBuildInfoLogSize);
        printf("Compiler response: %s", buildInfoLog);
        free(buildInfoLog);
    }

    cl_kernel kernel = clCreateKernel(program, "prefix_sum", &ret);
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&max_sum_mem_obj);

    cl_kernel kernel1 = clCreateKernel(program, "total_prefix_sum", &ret);
    ret = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)&b_mem_obj);
    ret = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&max_sum_mem_obj);
   
    cl_event kernel_event;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &kernel_event);
    ret = clEnqueueNDRangeKernel(command_queue, kernel1, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    float* B = new float[n];
    ret = clEnqueueReadBuffer(command_queue, b_mem_obj, CL_TRUE, 0, n * sizeof(float), B, 0, NULL, NULL);

    steady_clock::time_point end = high_resolution_clock::now();
    duration<double, std::milli> ms_double = end - start;

    cl_ulong begin_k = NULL, end_k = NULL;
    cl_ulong total_time = 0;
    ret = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &begin_k, NULL);
    ret = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_k, NULL);
    total_time += end_k - begin_k;

    for (int i = 0; i < n; i++)
        printf("%f\n", B[i]);

    printf("\nTime: %f\t%f \n", total_time * 1.0 / 1000000, ms_double.count());

    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    delete [] A;
    delete [] B;
    delete [] source_str;
    return 0;
}