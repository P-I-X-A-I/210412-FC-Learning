#include "LayerClass.h"

void LayerClass::init_as_fc_CEMS(int inW, int mode)
{
	// layer type string
	strcpy_s(layer_type_str, 32, "fcCEMS");

	input_width = inW;
	output_width = 1;

	fc_CEMS_mode = mode;

	const char* tempStr[2];
	tempStr[0] = "cross E";
	tempStr[1] = "mean S";

	if (isLog) { printf("init Layer as [%s (%s)]\n", layer_type_str, tempStr[mode]); }

	this->alloc_fc_CEMS_CPU_memory();
}


void LayerClass::alloc_fc_CEMS_CPU_memory()
{
	long dataSize;

	dataSize = sizeof(float) * 1;
	output_data_ptr = (float*)malloc(dataSize);
	memset(output_data_ptr, 0, dataSize);

	dataSize = sizeof(float)*input_width;
	back_out_ptr = (float*)malloc(dataSize);
	memset(back_out_ptr, 0, dataSize);

	dataSize = sizeof(float)*1;
	fc_CEMS_result_loss_sum = (float*)malloc(dataSize);
	*fc_CEMS_result_loss_sum = 0.0;
}


void LayerClass::alloc_fc_CEMS_cl_mem()
{
	cl_int error;
	// input data  // prev mem

	// output data
	long dataSize = sizeof(float) * num_GPU_image;
	error = this->create_mem_util(&mem_output_data, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 20\n"); }

	// back-in delta // prev mem

	// back-out mem
	dataSize = sizeof(float) * input_width * num_GPU_image;
	error = this->create_mem_util(&mem_back_out, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 21\n"); }

	// loss sum
	dataSize = sizeof(float) * 1;
	error = this->create_mem_util(&mem_fc_CEMS_loss_sum, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 22\n"); }

}





void LayerClass::input_data_to_fc_CEMS(float* inPtr, float* ansPtr)
{
	input_data_ptr = inPtr;

	// clear loss value ***
	*output_data_ptr = 0.0;


	switch (fc_CEMS_mode)
	{
	case 0: // cross entropy
		for (int i = 0; i < input_width; i++)
		{
			float inVal = *(input_data_ptr + i);
			float aVal = *(ansPtr + i);

			// log (0.0)is infinity, so add small value
			float entropy = -(aVal * log(inVal));

			// add to result( loss value )
			*output_data_ptr += entropy; // outW is 1

			// write back out
			*(back_out_ptr + i) = (-aVal) / inVal;
		}

		// sum loss value
		*fc_CEMS_result_loss_sum += *output_data_ptr;

		break;

	case 1: // mean square
		for (int i = 0; i < input_width; i++)
		{
			float inVal = *(input_data_ptr + i);
			float aVal = *(ansPtr + i);

			float meanS = (inVal - aVal)*(inVal - aVal);

			// add to result ( loss value )
			*output_data_ptr += meanS;

			// write back out
			*(back_out_ptr + i) = (inVal - aVal);
		}

		*output_data_ptr *= 0.5;

		// sum loss value
		*fc_CEMS_result_loss_sum += *output_data_ptr;

		break;

	default:
		break;
	}
}



void LayerClass::setup_fc_CEMS_kernel()
{
	// kernel
	this->create_kernel_util(&krn_fc_crossE, "fc_crossE");
	// arg
	this->set_kernel_arg_mem(&krn_fc_crossE, 0, &mem_input_data);
	this->set_kernel_arg_mem(&krn_fc_crossE, 1, &mem_answer);
	this->set_kernel_arg_mem(&krn_fc_crossE, 2, &mem_output_data);
	this->set_kernel_arg_val(&krn_fc_crossE, 3, answer_width);
	this->set_kernel_arg_mem(&krn_fc_crossE, 4, &mem_back_out);

	// kernel
	this->create_kernel_util(&krn_fc_meanS, "fc_meanS");
	// arg
	this->set_kernel_arg_mem(&krn_fc_meanS, 0, &mem_input_data);
	this->set_kernel_arg_mem(&krn_fc_meanS, 1, &mem_answer);
	this->set_kernel_arg_mem(&krn_fc_meanS, 2, &mem_output_data);
	this->set_kernel_arg_val(&krn_fc_meanS, 3, answer_width);
	this->set_kernel_arg_mem(&krn_fc_meanS, 4, &mem_back_out);

	
	// kernel
	this->create_kernel_util(&krn_fc_CEMS_sum_loss, "fc_CEMS_sum_loss");
	// arg
	this->set_kernel_arg_mem(&krn_fc_CEMS_sum_loss, 0, &mem_output_data);
	this->set_kernel_arg_mem(&krn_fc_CEMS_sum_loss, 1, &mem_fc_CEMS_loss_sum);
	this->set_kernel_arg_val(&krn_fc_CEMS_sum_loss, 2, num_GPU_image);
	
}


void LayerClass::run_fc_CEMS_kernel()
{
	cl_int error;
	size_t off_2D[2] = { 0, 0 };
	size_t work_2D[2] = { 1, num_GPU_image };
	size_t local_2D[2] = { 1, 10 };

	switch (fc_CEMS_mode)
	{
	case 0:// cross E
		error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
			krn_fc_crossE,
			2,
			off_2D, work_2D, local_2D,
			0, NULL, NULL);
		if (error != CL_SUCCESS) { printf("[krn_crossE] fail....%d\n", error); }
		break;

	case 1: // mean S
		error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
			krn_fc_meanS,
			2,
			off_2D, work_2D, local_2D,
			0, NULL, NULL);
		if (error != CL_SUCCESS) { printf("[krn_meanS] fail....%d\n", error); }
		break;
	}
}


void LayerClass::run_fc_CEMS_back_kernel()
{
	// sum up result loss

	cl_int error;
	size_t off_1D = 0;
	size_t work_1D = 1;
	size_t local_1D = 1;
	
	error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
		krn_fc_CEMS_sum_loss,
		1,
		&off_1D, &work_1D, &local_1D,
		0, NULL, NULL);

	if (error != CL_SUCCESS) { printf("[krn_fc_CEMS_sum_loss] fail....%d\n", error); }
	
}

void LayerClass::readback_CEMS()
{
	cl_int error;

	error = clEnqueueReadBuffer(
		cl_obj->cl_CMQ_obj[0],
		mem_fc_CEMS_loss_sum,
		CL_FALSE, 0,
		sizeof(float)*1, // loss value * numImage
		fc_CEMS_result_loss_sum,
		0, NULL, NULL
	);
}