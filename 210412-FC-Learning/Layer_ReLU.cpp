#include "LayerClass.h"



void LayerClass::init_as_fc_ReLU(int inW, int outW, int iniWeightMode)
{
	// layer type string
	strcpy_s(layer_type_str, 32, "fcReLU");

	if (isLog) { printf("init Layer as [%s]\n", layer_type_str); }

	input_width = inW;
	output_width = outW;

	// alloc memory for CPU size
	this->alloc_fc_ReLU_CPU_memory(iniWeightMode);
}


void LayerClass::alloc_fc_ReLU_CPU_memory(int iniMode)
{
	long dataSize;

	// out data
	dataSize = sizeof(float)*output_width;
	output_data_ptr = (float*)malloc(dataSize);
	memset(output_data_ptr, 0, dataSize);

	// weight
	dataSize = sizeof(float)*output_width * input_width;
	fc_ReLU_weight_ptr = (float*)malloc(dataSize);
	memset(fc_ReLU_weight_ptr, 0, dataSize);

	// bias
	dataSize = sizeof(float)*output_width;
	fc_ReLU_bias_ptr = (float*)malloc(dataSize);
	memset(fc_ReLU_bias_ptr, 0, dataSize);

	//////////////////
	// back out
	dataSize = sizeof(float)*input_width;
	back_out_ptr = (float*)malloc(dataSize);
	memset(back_out_ptr, 0, dataSize);

	// bias delta sum
	dataSize = sizeof(float)*output_width;
	fc_ReLU_bias_delta_sum = (float*)malloc(dataSize);
	memset(fc_ReLU_bias_delta_sum, 0, dataSize);

	dataSize = sizeof(float)*output_width;
	fc_ReLU_bias_velocity = (float*)malloc(dataSize);
	memset(fc_ReLU_bias_velocity, 0, dataSize);

	// weight delta sum
	dataSize = sizeof(float)*input_width * output_width;
	fc_ReLU_weight_delta_sum = (float*)malloc(dataSize);
	memset(fc_ReLU_weight_delta_sum, 0, dataSize);
	
	// weight velocity
	dataSize = sizeof(float)*input_width * output_width;
	fc_ReLU_weight_velocity = (float*)malloc(dataSize);
	memset(fc_ReLU_weight_velocity, 0, dataSize);





	/////////////////////////
	// generate initial weight
	this->generate_initial_HX_weight(iniMode);
}



void LayerClass::alloc_fc_ReLU_cl_mem()
{
	cl_int error;
	// input data  // prev mem

	// output data
	long dataSize = sizeof(float) * input_width * num_GPU_image;
	error = this->create_mem_util(&mem_output_data, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 10\n"); }

	// back-in delta // prev mem

	// back-out mem
	dataSize = sizeof(float) * input_width * num_GPU_image;
	error = this->create_mem_util(&mem_back_out, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 11\n"); }

	// weight // copy from cpu_weight ptr
	dataSize = sizeof(float) * input_width * output_width;
	mem_fc_ReLU_weight = clCreateBuffer(cl_obj->cl_CTX_obj[0],
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		dataSize,
		fc_ReLU_weight_ptr,
		&error);
	if (error != CL_SUCCESS) { printf("create mem error 12\n"); }

	// bias // copy from cpu bias ptr
	dataSize = sizeof(float)*output_width;
	mem_fc_ReLU_bias = clCreateBuffer(cl_obj->cl_CTX_obj[0],
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		dataSize,
		fc_ReLU_bias_ptr,
		&error);
	if (error != CL_SUCCESS) { printf("create mem error 13\n"); }

	// bias delta sum
	dataSize = sizeof(float)*output_width;
	error = this->create_mem_util(&mem_fc_ReLU_bias_delta_sum, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 14\n"); }

	// weight delta sum
	dataSize = sizeof(float)* input_width * output_width;
	error = this->create_mem_util(&mem_fc_ReLU_weight_delta_sum, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 15\n"); }

	// bias delta (for each image )
	dataSize = sizeof(float)*output_width * num_GPU_image;
	error = this->create_mem_util(&mem_fc_ReLU_bias_delta, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 16\n"); }

	// weight delta ( for each image )
	dataSize = sizeof(float) * input_width * output_width * num_GPU_image;
	error = this->create_mem_util(&mem_fc_ReLU_weight_delta, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 17\n"); }


}




void LayerClass::generate_initial_HX_weight(int iniMode)
{
	if (strcmp("fcReLU", layer_type_str) == 0)
	{
		float coef = 1.0;
		if (iniMode == 0) { coef = 2.0; }

		// weight initialize
		float deviation = coef / sqrt((float)input_width);

		// C++ random
		std::random_device rDevice; // normal random device
		std::mt19937 mersenne(rDevice()); // 32bit Mersenne Twister
		std::normal_distribution<> norm_dist(0.0, deviation); // 

		// write weight
		float* writePtr = fc_ReLU_weight_ptr;

		for (int n = 0; n < output_width; n++)
		{
			for (int w = 0; w < input_width; w++)
			{
				*writePtr = norm_dist(mersenne);
				writePtr++;
			}
		}

		// bias initialize
		float* writeBiasPtr = fc_ReLU_bias_ptr;
		for (int n = 0; n < output_width; n++)
		{
			*writeBiasPtr = norm_dist(mersenne);
			writeBiasPtr++;
		}
	}
}




void LayerClass::input_data_to_fc_ReLU(float* inPtr)
{
	input_data_ptr = inPtr;

	// calc weight
	for (int n = 0; n < output_width; n++)
	{
		float FIRE = 0.0;
		float* weightHead = fc_ReLU_weight_ptr + (input_width * n);

		// calc (input * weight)
		for (int w = 0; w < input_width; w++)
		{
			FIRE += (*(weightHead + w)) * (*(input_data_ptr + w));
		}

		// add bias
		FIRE += *(fc_ReLU_bias_ptr + n);

		// ReLU filter
		FIRE = fmaxf(0.0, FIRE);

		// write to output
		*(output_data_ptr + n) = FIRE;
	}
}


void LayerClass::back_propagation_fc_ReLU(float* backInPtr)
{
	back_in_ptr = backInPtr;

	// bias delta is back-in value itself

	for (int i = 0; i < output_width; i++)
	{
		float mask = 1.0;
		float prevOut = *(output_data_ptr + i);
		float backValue = *(back_in_ptr + i);

		if (prevOut == 0.0)
		{
			mask = 0.0;
		}

		// add bias sum
		*(fc_ReLU_bias_delta_sum + i) += mask * backValue;
	}


	// weight delta is (back-in) * (prev_input)
	for (int n = 0; n < output_width; n++)
	{
		float mask = 1.0;
		if (*(output_data_ptr + n) == 0.0)
		{
			mask = 0.0;
		}

		float backVal = (*(back_in_ptr + n)) * mask;

		// add weight sum
		float* weiDelPtr = fc_ReLU_weight_delta_sum + (n * input_width);
		
		for (int w = 0; w < input_width; w++)
		{
			float inputVal = *(input_data_ptr + w);
			*(weiDelPtr + w) += backVal * inputVal;
		}
	}


	// back out value
	for (int w = 0; w < input_width; w++)
	{
		float backSum = 0.0;

		for (int n = 0; n < output_width; n++)
		{
			float backInVal = *(back_in_ptr + n);
			float weightVal = *(fc_ReLU_weight_ptr + (n*input_width) + w);

			backSum += (backInVal * weightVal);
		}

		*(back_out_ptr + w) = backSum;
	}

}




void LayerClass::setup_fc_ReLU_kernel()
{
	// kernel
	this->create_kernel_util(&krn_fc_ReLU, "fc_ReLU");
	// set arg
	this->set_kernel_arg_mem(&krn_fc_ReLU, 0, &mem_input_data);
	this->set_kernel_arg_mem(&krn_fc_ReLU, 1, &mem_fc_ReLU_weight);
	this->set_kernel_arg_mem(&krn_fc_ReLU, 2, &mem_fc_ReLU_bias);
	this->set_kernel_arg_mem(&krn_fc_ReLU, 3, &mem_output_data);
	this->set_kernel_arg_val(&krn_fc_ReLU, 4, input_width);

	// kernel
	this->create_kernel_util(&krn_fc_ReLU_back_biasDelta, "fc_ReLU_back_biasDelta");
	// arg
	this->set_kernel_arg_mem(&krn_fc_ReLU_back_biasDelta, 0, &mem_output_data);
	this->set_kernel_arg_mem(&krn_fc_ReLU_back_biasDelta, 1, &mem_fc_ReLU_bias_delta);
	this->set_kernel_arg_mem(&krn_fc_ReLU_back_biasDelta, 2, &mem_back_in);
	this->set_kernel_arg_val(&krn_fc_ReLU_back_biasDelta, 3, output_width);

	// kernel
	this->create_kernel_util(&krn_fc_ReLU_sum_biasDelta, "fc_ReLU_sum_biasDelta");
	// arg
	this->set_kernel_arg_mem(&krn_fc_ReLU_sum_biasDelta, 0, &mem_fc_ReLU_bias_delta);
	this->set_kernel_arg_mem(&krn_fc_ReLU_sum_biasDelta, 1, &mem_fc_ReLU_bias_delta_sum);
	this->set_kernel_arg_val(&krn_fc_ReLU_sum_biasDelta, 2, num_GPU_image);

	// kernel
	this->create_kernel_util(&krn_fc_ReLU_back_weightDelta, "fc_ReLU_back_weightDelta");
	// arg
	this->set_kernel_arg_mem(&krn_fc_ReLU_back_weightDelta, 0, &mem_back_in);
	this->set_kernel_arg_mem(&krn_fc_ReLU_back_weightDelta, 1, &mem_input_data);
	this->set_kernel_arg_mem(&krn_fc_ReLU_back_weightDelta, 2, &mem_fc_ReLU_weight_delta);
	this->set_kernel_arg_val(&krn_fc_ReLU_back_weightDelta, 3, output_width);

	// kernel
	this->create_kernel_util(&krn_fc_ReLU_sum_weightDelta, "fc_ReLU_sum_weightDelta");
	// arg
	this->set_kernel_arg_mem(&krn_fc_ReLU_sum_weightDelta, 0, &mem_fc_ReLU_weight_delta);
	this->set_kernel_arg_mem(&krn_fc_ReLU_sum_weightDelta, 1, &mem_fc_ReLU_weight_delta_sum);
	this->set_kernel_arg_val(&krn_fc_ReLU_sum_weightDelta, 2, num_GPU_image);


	// kernel 
	this->create_kernel_util(&krn_fc_ReLU_back_out, "fc_ReLU_back_out");
	// arg
	this->set_kernel_arg_mem(&krn_fc_ReLU_back_out, 0, &mem_back_in);
	this->set_kernel_arg_mem(&krn_fc_ReLU_back_out, 1, &mem_fc_ReLU_weight);
	this->set_kernel_arg_mem(&krn_fc_ReLU_back_out, 2, &mem_back_out);
	this->set_kernel_arg_val(&krn_fc_ReLU_back_out, 3, output_width);


}


void LayerClass::run_fc_ReLU_kernel()
{
	cl_int error;
	size_t off_2D[2] = { 0, 0 };
	size_t work_2D[2] = { output_width, num_GPU_image};
	size_t local_2D[2] = {2, 50};
	
	error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
		krn_fc_ReLU,
		2,
		off_2D, work_2D, local_2D,
		0, NULL, NULL);
	if (error != CL_SUCCESS) { printf("[krn_ReLU] fail....%d\n", error); }
	
}



void LayerClass::run_fc_ReLU_back_kernel()
{
	cl_int error;
	size_t off_2D[2] = { 0, 0 };
	size_t work_2D[2] = { 1, num_GPU_image };
	size_t local_2D[2] = { 1, 50 };

	// calc until bias delta & updata back-in by mask
	
	error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
		krn_fc_ReLU_back_biasDelta,
		2,
		off_2D, work_2D, local_2D,
		0, NULL, NULL);

	if (error != CL_SUCCESS) { printf("[krn_ReLU_back_biasDelta] fail....%d\n", error); }
	
	// sum bias delta/////////////////////////////////////
	work_2D[0] = 1;
	work_2D[1] = output_width;
	local_2D[0] = 1;
	local_2D[1] = 2;

	// sum_bias Delta
	
	error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
		krn_fc_ReLU_sum_biasDelta,
		2,
		off_2D, work_2D, local_2D,
		0, NULL, NULL);

	if (error != CL_SUCCESS) { printf("[krn_ReLU_sum_biasDelta] fail....%d\n", error); }
	
	// weight delta ////////////////////////////////////
	
	
	size_t off_3D[3] = {0, 0, 0};
	size_t work_3D[3] = { input_width, output_width, num_GPU_image };
	size_t local_3D[3] = { 2, 2, 50 };
	
	// weight Delta
	
	/*
	work_2D[0] = input_width;
	work_2D[1] = num_GPU_image;
	local_2D[0] = 2;
	local_2D[1] = 50;
	*/
	
	error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
		krn_fc_ReLU_back_weightDelta,
		3,
		off_3D, work_3D, local_3D,
		0, NULL, NULL);
	
	if (error != CL_SUCCESS) { printf("[krn_ReLU_back_weightDelta] fail....%d\n", error); }
	

	// sum weight delta
	work_2D[0] = input_width;
	work_2D[1] = output_width;
	local_2D[0] = 2;
	local_2D[1] = 2;
	
	error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
		krn_fc_ReLU_sum_weightDelta,
		2,
		off_2D, work_2D, local_2D,
		0, NULL, NULL);

	if (error != CL_SUCCESS) { printf("[krn_ReLU_sum_weightDelta] fail....%d\n", error); }
	

	// back-out
	work_2D[0] = input_width;
	work_2D[1] = num_GPU_image;
	local_2D[0] = 2;
	local_2D[1] = 50;

	error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
		krn_fc_ReLU_back_out,
		2,
		off_2D, work_2D, local_2D,
		0, NULL, NULL);

	if (error != CL_SUCCESS) { printf("[krn_ReLU_back_out] fail....%d\n", error); }

}



void LayerClass::readback_ReLU()
{
	cl_int error;

	// read back bias delta sum
	error = clEnqueueReadBuffer(
		cl_obj->cl_CMQ_obj[0],
		mem_fc_ReLU_bias_delta_sum,
		CL_FALSE, 0,
		sizeof(float)*output_width, 
		fc_ReLU_bias_delta_sum,
		0, NULL, NULL
	);

	// read back weight delta sum
	error = clEnqueueReadBuffer(
		cl_obj->cl_CMQ_obj[0],
		mem_fc_ReLU_weight_delta_sum,
		CL_FALSE, 0,
		sizeof(float)*input_width*output_width, 
		fc_ReLU_weight_delta_sum,
		0, NULL, NULL
	);

}