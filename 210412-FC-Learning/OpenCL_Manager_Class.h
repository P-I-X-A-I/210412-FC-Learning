#pragma once

#include "framework.h"

// OpenCL
#pragma comment(lib, "CL/x64/OpenCL.lib")
//#pragma comment(lib, "CL/x86/OpenCL.lib")
#include "CL/cl.h"

#include <fstream>

#define NUM_MAX_CL_DEVICE 8

class OpenCL_Manager_Class
{
public:
	cl_platform_id platformID[64];
	///////////////////////////////
	cl_device_id GPUdeviceID_array[NUM_MAX_CL_DEVICE];
	cl_uint num_devices;
	///////////////////////////////
	cl_context cl_CTX_obj[NUM_MAX_CL_DEVICE];
	cl_command_queue cl_CMQ_obj[NUM_MAX_CL_DEVICE];
	cl_program cl_PRG_obj[32];

	// device info
	cl_uint num_compute_unit[NUM_MAX_CL_DEVICE];
	int max_work_in_group[NUM_MAX_CL_DEVICE][3];
	int local_size[NUM_MAX_CL_DEVICE];



	OpenCL_Manager_Class();
	~OpenCL_Manager_Class();

	///////// set ///////////////////
	void setup_openCL();
	void create_program_from_file(int ctxIDX, int prgIDX, const char* filePath);

private:
	/////////////////////////////////
	void setup_platform();
	void setup_gpu_device();
	void get_device_info_func(int devIDX, int ENUM, long size, void* ptr);
	void create_context_and_queue();
	////////////////////////////
};

