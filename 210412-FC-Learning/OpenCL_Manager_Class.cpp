#include "OpenCL_Manager_Class.h"


OpenCL_Manager_Class::OpenCL_Manager_Class()
{
	printf("OpenCL Manager Class init\n");

}


OpenCL_Manager_Class::~OpenCL_Manager_Class()
{

}

void OpenCL_Manager_Class::setup_openCL()
{
	this->setup_platform();
	this->setup_gpu_device();
	this->create_context_and_queue();
}

void OpenCL_Manager_Class::setup_platform()
{
	// get platform ID
	cl_int status;
	cl_uint num_platform;
	status = clGetPlatformIDs(64,
		platformID,
		&num_platform);

	if (status == CL_SUCCESS) { printf("\n\n*** get OpenCL platform ID SUCCESS ***\n"); }

	for (int i = 0; i < num_platform; i++)
	{
		// get platform info /////////////////////////
		char Str[2048];
		size_t rsz;

		// platform profile
		status = clGetPlatformInfo(platformID[i], CL_PLATFORM_PROFILE, 512, Str, &rsz);
		if (status == CL_SUCCESS) { printf("* platform profile(%d) : %s\n",i, Str); }

		status = clGetPlatformInfo(platformID[i], CL_PLATFORM_VERSION, 512, Str, &rsz);
		if (status == CL_SUCCESS) { printf("* version(%d) : %s\n", i, Str); }

		status = clGetPlatformInfo(platformID[i], CL_PLATFORM_NAME, 512, Str, &rsz);
		if (status == CL_SUCCESS) { printf("* name(%d) : %s\n", i, Str); }

		status = clGetPlatformInfo(platformID[i], CL_PLATFORM_VENDOR, 512, Str, &rsz);
		if (status == CL_SUCCESS) { printf("* vendor(%d) : %s\n", i, Str); }

		status = clGetPlatformInfo(platformID[i], CL_PLATFORM_EXTENSIONS, 512, Str, &rsz);
		if (status == CL_SUCCESS) { printf("* extensions(%d) : %s\n", i, Str, &rsz); }
	}
}

void OpenCL_Manager_Class::setup_gpu_device()
{
	// find all devices
	cl_int status;
	size_t rsz;
	cl_device_type dev_type;

	// search all devices
	status = clGetDeviceIDs(
		platformID[0], 
		CL_DEVICE_TYPE_GPU,
		NUM_MAX_CL_DEVICE,// array size 
		GPUdeviceID_array,
		&num_devices);

	if (status == CL_SUCCESS)
	{
		printf("\n*** num (%d) GPU device found in platform[0] ***\n", num_devices);
		printf("---(always use platform[0] in this class)---\n");
	}


	for (int i = 0; i < num_devices; i++)
	{
		// vendor id
		char vendor[256];
		this->get_device_info_func(i, CL_DEVICE_VENDOR, 256, vendor);
		printf("* device vendor(%d) : %s\n", i, vendor);

		// device name
		char dev_name[256];
		this->get_device_info_func(i, CL_DEVICE_NAME, 256, dev_name);
		printf("* device name(%d) : %s\n", i, dev_name);

		// openCL version
		char version[128];
		this->get_device_info_func(i, CL_DEVICE_OPENCL_C_VERSION, 128, version);
		printf("* CL version(%d) : %s\n", i, version);

		// device extension
		char dev_ext[2048];
		this->get_device_info_func(i, CL_DEVICE_EXTENSIONS, 2048, dev_ext);
		printf("* device ext(%d) : %s\n", i, dev_ext);

		// max mem size
		cl_ulong memsize;
		this->get_device_info_func(i, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &memsize);
		printf("* max mem alloc size(%d) : %dMB\n", i, memsize/1024/1024);

		// freq
		cl_uint gpu_freq;
		this->get_device_info_func(i, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &gpu_freq);
		printf("* GPU frequency(%d) :  %d Mhz\n",i,  gpu_freq);

		// local memory size
		cl_ulong localMemSize;
		this->get_device_info_func(i, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize);
		printf("* Local memory size (%d) : %d KB\n", i, localMemSize / 1024);

		// local memory type
		cl_device_local_mem_type localMemType;
		this->get_device_info_func(i, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &localMemType);
		printf("* Local memory type (%d) : %d [1=CL_LOCAL, 2=CL_GLOBAL]\n", i, localMemType); 

		// support for fine grained buffer
		cl_device_svm_capabilities capa;
		this->get_device_info_func(i, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &capa);
		printf("* support for SVM (%d) : %d []\n", i, capa); 

		//////// important /////////////
		cl_uint num_cu;
		this->get_device_info_func(i, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cu);
		printf("* compute unit(%d) : %d\n", i, num_cu);
		///////////////////////////////
		num_compute_unit[i] = num_cu;
		///////////////////////////////

		// max dimension
		cl_uint max_dim;
		this->get_device_info_func(i, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_dim);
		printf("* max dimension(%d) : %d\n", i, max_dim);

		// max address bit ( max global work size )
		cl_uint add_size;
		this->get_device_info_func(i, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &add_size);
		printf("* address bit(%d) : %d bit\n", i, add_size);

		// work size[x][y][z]
		size_t w_size[3];
		this->get_device_info_func(i, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(w_size), w_size);
		printf("* work size(%d) : [%d][%d][%d]\n", i, w_size[0], w_size[1], w_size[2]);
		////////////////////////////////
		max_work_in_group[i][0] = w_size[0];
		max_work_in_group[i][1] = w_size[1];
		max_work_in_group[i][2] = w_size[2];
		////////////////////////////////

		// max local size
		size_t maxLocal;
		this->get_device_info_func(i, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxLocal), &maxLocal);
		printf("* local (item in group) size(%d) :  %d\n", i, maxLocal);

		////////////////////////////
		local_size[i] = maxLocal;
		////////////////////////////

	} // for i < dev num
}

void OpenCL_Manager_Class::get_device_info_func(int devIDX, 
	int ENUM,
	long size,
	void* ptr)
{
	cl_int status;
	size_t rsz;

	status = clGetDeviceInfo(
	GPUdeviceID_array[devIDX],
		ENUM,
		size,
		ptr,
		&rsz);
}


void OpenCL_Manager_Class::create_context_and_queue()
{
	for (int i = 0; i < num_devices; i++)
	{
		cl_int error;

		cl_CTX_obj[i] = clCreateContext(
			NULL, // property list
			1, // num device
			&GPUdeviceID_array[i],
			NULL, // callback for error
			NULL, // object passed to callback
			&error
		);
	
		if (error == CL_SUCCESS) { printf("\n* create openCL context for device(%d) SUCCESS!\n", i, i); }


		//////////////////////
		// create command queue

		// for OpenCL 1.2
		/*
		cl_CMQ_obj[i] = clCreateCommandQueue(
		cl_CTX_obj[i],
		deviceID_array[i],
		CL_QUEUE_PROFILING_ENABLE,
		&error);
		*/

		// for OpenCL 2.0
		cl_command_queue_properties prop[3] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };

		cl_CMQ_obj[i] = clCreateCommandQueueWithProperties(
		cl_CTX_obj[i],
			GPUdeviceID_array[i],
			prop,
			&error);

		if (error == CL_SUCCESS) { printf("* create command queue(%d) SUCCESS!\n", i); }
	}
}


//////////////////////////////////////////

void OpenCL_Manager_Class::create_program_from_file(int ctxIDX, int prgIDX, const char* filePath)
{
	// get text data
	std::ifstream sFile;
	sFile.open(filePath);

	if (!sFile)
	{
		printf("open file %s fail... return\n", filePath);
		return;
	}

	// set file's position to end
	sFile.seekg(0, std::ios::end);

	// get file length
	unsigned long fLength = sFile.tellg();

	// reset file position
	sFile.seekg(std::ios::beg);

	// create array for kernel source
	char* sourceArray = new char[fLength + 1];

	// set terminate zero
	sourceArray[fLength] = 0;

	// read source string
	size_t INDEX = 0;
	while (sFile.good())
	{
		sourceArray[INDEX] = sFile.get();
		if (!sFile.eof())
		{
			INDEX++;
		}
	}

	// set terminate zero again
	sourceArray[INDEX] = 0;

	// close file
	sFile.close();

	//::::::::::::::::::::::::::::::::::

	// craet program
	cl_int error;

	cl_PRG_obj[prgIDX] = clCreateProgramWithSource(
	cl_CTX_obj[ctxIDX],
		1, // num source
		(const char**)&sourceArray,
		&INDEX, // source size
		&error);

	if (error == CL_SUCCESS) { printf("[%s] create program(%d) with context(%d) SUCCESS!\n", filePath, prgIDX, ctxIDX); }


	/////////////////////////////////////////
	//// build program

	const char* option = "-cl-std=CL2.0";

	// compile & link
	error = clBuildProgram(
	cl_PRG_obj[prgIDX],
		1,
		&GPUdeviceID_array[ctxIDX],
		option,
		NULL,
		NULL);

	if (error == CL_SUCCESS) { printf("build OpenCL program(%d) in context(%d) SUCCESS!\n", prgIDX, ctxIDX); }
	else
	{
		char logString[50000];
		size_t rSize;
		clGetProgramBuildInfo(
			cl_PRG_obj[prgIDX],
			GPUdeviceID_array[ctxIDX],
			CL_PROGRAM_BUILD_LOG,
			50000,
			logString,
			&rSize);

		printf("\n\n[build error!!!!!!] program build log %d\n\n%s\n", rSize, logString);
	}
}

