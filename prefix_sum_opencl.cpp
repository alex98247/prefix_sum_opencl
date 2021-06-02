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

#define BUFFER_SIZE (0x100000)

int main(int argc, char* argv[]) {

    if (argc != 4) {
        std::cerr << "Invalid args count" << std::endl;
        return 1;
    }

    int global_id_device = atoi(argv[1]);
    if ((!global_id_device && strcmp(argv[1], "0") != 0) || global_id_device < -1) {
        std::cerr << "Invalid threads count arg" << std::endl;
        return 1;
    }
    
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    std::vector <cl_device_id> cpu_devices;
    std::vector <cl_device_id> gpu_descrete_devices;
    std::vector <cl_device_id> gpu_nondescrete_devices;
    cl_ulong device_host_unified_memory;
    cl_device_type device_type;
    cl_int err;

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

    std::string file_path_in = argv[2];
    int n = 0;
    float* A;
    std::ifstream in(file_path_in);

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
        std::cerr << "Can not read file " << file_path_in << std::endl;
        return 1;
    }
    in.close();

    FILE* fp;
    char* source_str;
    size_t source_size;

    fopen_s(&fp, "prefix_sum_kernel.cl", "r");
    if (!fp) {
        std::cerr << "Failed to load kernel " << "prefix_sum_kernel.cl" << std::endl;
        delete[] A;
        return 1;
    }
    source_str = new char [BUFFER_SIZE];
    if (!source_str) {
        std::cerr << "Can not allocate memory" << std::endl;
        delete[] A;
        delete[] source_str;
        return 1;
    }
    source_size = fread(source_str, 1, BUFFER_SIZE, fp);
    fclose(fp);


    cl_device_id* device_id = NULL;
    if (global_id_device < 0) {
        std::cerr << "Invalid device id" << std::endl;
        delete[] source_str;
        delete[] A;
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
        std::cerr << "Invalid device id" << std::endl;
        delete[] source_str;
        delete[] A;
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
    cl_context context = clCreateContext(NULL, 1, device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Can not create context" << std::endl;
        delete[] source_str;
        delete[] A;
        return 1;
    }

    cl_command_queue command_queue = clCreateCommandQueue(context, *device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Can not create command queue" << std::endl;
        delete[] source_str;
        delete[] A;
        return 1;
    }

    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Can not create buffer" << std::endl;
        delete[] source_str;
        delete[] A;
        return 1;
    }
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Can not create buffer" << std::endl;
        delete[] source_str;
        delete[] A;
        return 1;
    }
    cl_mem max_sum_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, global_groups_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Can not create buffer" << std::endl;
        delete[] source_str;
        delete[] A;
        return 1;
    }

    if (clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, n * sizeof(float), A, 0, NULL, NULL) != CL_SUCCESS) {
        std::cerr << "Can not write buffer" << std::endl;
        delete[] source_str;
        delete[] A;
        return 1;
    }

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Can not create program" << std::endl;
        delete[] source_str;
        delete[] A;
        return 1;
    }

    delete[] source_str;
    if (clBuildProgram(program, 1, device_id, NULL, NULL, NULL) != CL_SUCCESS) {
        std::cerr << "Can not create buffer" << std::endl;
        delete[] A;
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "prefix_sum", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Can not create kernel" << std::endl;
        delete[] A;
        return 1;
    }
    if (clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj) != CL_SUCCESS) {
        std::cerr << "Can not set argument" << std::endl;
        delete[] A;
        return 1;
    }
    if (clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj) != CL_SUCCESS) {
        std::cerr << "Can not set argument" << std::endl;
        delete[] A;
        return 1;
    }
    if (clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&max_sum_mem_obj) != CL_SUCCESS) {
        std::cerr << "Can not set argument" << std::endl;
        delete[] A;
        return 1;
    }

    cl_kernel kernel1 = clCreateKernel(program, "total_prefix_sum", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Can not create kernel" << std::endl;
        delete[] A;
        return 1;
    }
    if (clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)&b_mem_obj) != CL_SUCCESS) {
        std::cerr << "Can not set argument" << std::endl;
        delete[] A;
        return 1;
    }
    if (clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&max_sum_mem_obj) != CL_SUCCESS) {
        std::cerr << "Can not set argument" << std::endl;
        delete[] A;
        return 1;
    }
   
    cl_event kernel_event, kernel_event1;
    if (clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &kernel_event) != CL_SUCCESS) {
        std::cerr << "Can not start kernel" << std::endl;
        delete[] A;
        return 1;
    }
    if (clEnqueueNDRangeKernel(command_queue, kernel1, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &kernel_event1) != CL_SUCCESS) {
        std::cerr << "Can not start kernel" << std::endl;
        delete[] A;
        return 1;
    }

    float* B = new float[n];
    if (!B) {
        std::cerr << "Can not allocate memory" << std::endl;
        delete[] B;
        delete[] A;
        return 1;
    }
    if (clEnqueueReadBuffer(command_queue, b_mem_obj, CL_TRUE, 0, n * sizeof(float), B, 0, NULL, NULL) != CL_SUCCESS) {
        std::cerr << "Can not read buffer " << err << std::endl;
        delete[] B;
        delete[] A;
        return 1;
    }

    steady_clock::time_point end = high_resolution_clock::now();
    duration<double, std::milli> ms_double = end - start;

    cl_ulong begin_k = NULL, end_k = NULL;
    cl_ulong total_time = 0;
    if (clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &begin_k, NULL) != CL_SUCCESS) {
        std::cerr << "Can not get profiling info" << std::endl;
        delete[] B;
        delete[] A;
        return 1;
    }
    if (clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_k, NULL) != CL_SUCCESS) {
        std::cerr << "Can not get profiling info" << std::endl;
        delete[] B;
        delete[] A;
        return 1;
    }
    total_time += end_k - begin_k;
    if (clGetEventProfilingInfo(kernel_event1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &begin_k, NULL) != CL_SUCCESS) {
        std::cerr << "Can not get profiling info" << std::endl;
        delete[] B;
        delete[] A;
        return 1;
    }
    if (clGetEventProfilingInfo(kernel_event1, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_k, NULL) != CL_SUCCESS) {
        std::cerr << "Can not get profiling info" << std::endl;
        delete[] B;
        delete[] A;
        return 1;
    }
    total_time += end_k - begin_k;

    std::string file_path_out = argv[3];
    std::ofstream out(file_path_out);
    if (out.is_open())
    {
        for (int i = 0; i < n; i++)
            out << B[i] << " ";
    }
    else
    {
        std::cerr << "Can not read file " << file_path_out << std::endl;
        return 1;
    }
    out.close();

    printf("\nTime: %f\t%f \n", total_time * 1.0 / 1000000, ms_double.count());

    delete[] A;
    delete[] B;
    if(clReleaseKernel(kernel) != CL_SUCCESS) {
        std::cerr << "Can not release kernel" << std::endl;
        return 1;
    }
    if (clReleaseKernel(kernel1) != CL_SUCCESS) {
        std::cerr << "Can not release kernel" << std::endl;
        return 1;
    }
    if (clReleaseProgram(program) != CL_SUCCESS) {
        std::cerr << "Can not release program" << std::endl;
        return 1;
    }
    if(clReleaseMemObject(a_mem_obj) != CL_SUCCESS) {
        std::cerr << "Can not release mem object" << std::endl;
        return 1;
    }
    if(clReleaseMemObject(b_mem_obj) != CL_SUCCESS) {
        std::cerr << "Can not release mem object" << std::endl;
        return 1;
    }
    if(clReleaseMemObject(max_sum_mem_obj) != CL_SUCCESS) {
        std::cerr << "Can not release mem object" << std::endl;
        return 1;
    }
    if(clReleaseCommandQueue(command_queue) != CL_SUCCESS) {
        std::cerr << "Can not release command queue" << std::endl;
        return 1;
    }
    if(clReleaseContext(context) != CL_SUCCESS) {
        std::cerr << "Can not release context" << std::endl;
        return 1;
    }
    return 0;
}