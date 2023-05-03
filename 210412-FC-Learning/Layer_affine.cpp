#include "LayerClass.h"

void LayerClass::init_as_fc_affine(int inW, int mode)
{
	// layer type string
	strcpy_s(layer_type_str, 32, "fcAff");

	input_width = inW;
	output_width = inW;
	fc_affine_mode = mode;

	const char* tempStr[3];
	tempStr[0] = "pass";
	tempStr[1] = "batch norm";
	tempStr[2] = "softmax";

	if (isLog) { printf("init Layer as [%s (%s)]\n", layer_type_str, tempStr[mode]); }

	this->alloc_fc_affine_CPU_memory();
}

//////////////////////////////////////////////////////

void LayerClass::alloc_fc_affine_CPU_memory()
{
	long dataSize;

	// out data
	dataSize = sizeof(float)*output_width;
	output_data_ptr = (float*)malloc(dataSize);
	memset(output_data_ptr, 0, dataSize);

	////////////////////////////////////////////
	// back out data
	dataSize = sizeof(float)*input_width;
	back_out_ptr = (float*)malloc(dataSize);
	memset(back_out_ptr, 0, dataSize);

	// hold deviation
	dataSize = sizeof(float);
	fc_affine_hold_deviation = (float*)malloc(dataSize);
	memset(fc_affine_hold_deviation, 0, dataSize);

	// hold exp
	dataSize = sizeof(float)*input_width;
	fc_affine_hold_exp = (float*)malloc(dataSize);
	memset(fc_affine_hold_exp, 0, dataSize);

	// hold expSum
	dataSize = sizeof(float);
	fc_affine_hold_expSum = (float*)malloc(dataSize);
	memset(fc_affine_hold_expSum, 0, dataSize);
}


void LayerClass::alloc_fc_affine_cl_mem()
{
	cl_int error;
	// input data  // prev mem

	// output data
	long dataSize = sizeof(float) * output_width * num_GPU_image;
	error = this->create_mem_util(&mem_output_data, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 0\n"); }

	//////////////////////////////////////////////////
	// back-in delta // prev mem

	// back-out mem
	dataSize = sizeof(float) * input_width * num_GPU_image;
	error = this->create_mem_util(&mem_back_out, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 1\n"); }
	
	// hold deviation
	dataSize = sizeof(float) * num_GPU_image;
	error = this->create_mem_util(&mem_fc_affine_hold_deviation, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 2\n"); }

	// hold exp each
	dataSize = sizeof(float) * input_width * num_GPU_image;
	error = this->create_mem_util(&mem_fc_affine_hold_exp, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 3\n"); }

	// hold expSum
	dataSize = sizeof(float) * num_GPU_image;
	error = this->create_mem_util(&mem_fc_affine_hold_expSum, dataSize, CL_MEM_READ_WRITE);
	if (error != CL_SUCCESS) { printf("create mem error 4\n"); }

	// temp variable for calculatio of backP batch norm
	dataSize = sizeof(float)*input_width*num_GPU_image;
	error = this->create_mem_util(&mem_fc_affine_temp_variable, dataSize, CL_MEM_READ_WRITE);

	if (error != CL_SUCCESS) { printf("create mem error 5\n"); }

}


///////////////////////////////////////////////

void LayerClass::input_data_to_fc_affine(float* inPtr)
{
	input_data_ptr = inPtr;

	// copy to output_data_ptr once
	long dataSize = sizeof(float)*output_width;
	memcpy(output_data_ptr, inPtr, dataSize);

	// data processing
	switch (fc_affine_mode)
	{
	case 0: // pass
		break;

	case 1: // batch norm
		this->batch_normalization();
		break;

	case 2: // softmax
		this->softmax();
		break;

	default:
		break;
	}
}


void LayerClass::batch_normalization()
{
	// calc average
	float ave = 0.0;
	float S = 0.0;

	for (int i = 0; i < output_width; i++)
	{
		S += *(output_data_ptr + i);
	}

	ave = S / (float)output_width;


	// calc deviation
	float DEVI = 0.0;

	for (int i = 0; i < output_width; i++)
	{
		float temp = (*(output_data_ptr + i)) - ave;
		DEVI += temp * temp;
	}

	DEVI /= (float)output_width;

	// hold deviation for back-propagation
	*fc_affine_hold_deviation = DEVI;


	// write to output_ptr
	for (int i = 0; i < output_width; i++)
	{
		float temp = *(output_data_ptr + i);
		*(output_data_ptr + i) = (temp - ave) / sqrt(DEVI + 0.0001);
	}

}

void LayerClass::softmax()
{
	// find max
	float M = 0.0;

	for (int i = 0; i < input_width; i++)
	{
		if (M < *(input_data_ptr + i))
		{
			M = *(input_data_ptr + i);
		}
	}
	

	// sumExp
	float sumExp = 0.0; 
	for (int i = 0; i < input_width; i++)
	{
		float expVal = exp(*(input_data_ptr + i) - M);
		sumExp += expVal;

		// hold each exp value for back-propagation
		*(fc_affine_hold_exp + i) = expVal;
	}

	// hold expSum for back-propagation
	*fc_affine_hold_expSum = sumExp;
	

	// write to output ptr
	for (int i = 0; i < output_width; i++)
	{
		float tempVal = *(input_data_ptr + i);
		*(output_data_ptr + i) = exp(tempVal - M) / sumExp; //*?*?*?***?*?*?*?*?*?*?*?
	}

}



void LayerClass::back_propagation_fc_affine(float* backInPtr)
{
	back_in_ptr = backInPtr;

	switch (fc_affine_mode)
	{
	case 0: // pass
		memcpy(back_out_ptr, back_in_ptr, sizeof(float)*input_width);
		break;

	case 1:
		this->back_batch_normalization();
		break;

	case 2:
		this->back_softmax();
		break;
	}
}


void LayerClass::back_batch_normalization()
{
	// A[0] = (back[0] / devi) - ( sqrt(devi)/devi^2 * (1.0/m) * prev[0] * sum(prev[k]*back[k]));
	// R[0] = A[0] + (1/m)sum(A[k])

	// alloc A
	float* A = (float*)malloc(sizeof(float)*input_width);

	// calc sum( backin[k] * prevOut[k] )
	float sumBack = 0.0;

	for (int i = 0; i < input_width; i++)
	{
		float backVal = *(back_in_ptr + i);
		float outVal = *(output_data_ptr + i);

		sumBack += backVal * outVal;
	}

	// calc each A
	float DEVI = *fc_affine_hold_deviation;

	// constant
	float C = sqrt(DEVI) / (DEVI*DEVI);
	float m = 1.0 / (float)input_width;

	for (int i = 0; i < input_width; i++)
	{
		float backVal = *(back_in_ptr + i);
		float outVal = *(output_data_ptr + i);
		float B = (backVal / DEVI);
		float D = outVal * sumBack;

		*(A + i) = B - ( C * m * D);
	}

	// sum A
	float sumA = 0.0;
	for (int i = 0; i < input_width; i++)
	{
		sumA += *(A + i);
	}

	sumA /= (float)input_width;


	// write to back-out
	for (int i = 0; i < input_width; i++)
	{
		*(back_out_ptr + i) = *(A + i) + sumA;
	}

	// free A
	free(A);
}


void LayerClass::back_softmax()
{
	// back = out[0]*back[0] - sum(out[k]*back[k]);

	float sumBack = 0.0;
	for (int i = 0; i < input_width; i++)
	{
		float prevExp = *(fc_affine_hold_exp + i);
		float backVal = *(back_in_ptr + i);

		sumBack += prevExp * backVal;
	}

	// -1.0 / S^2
	float S = *fc_affine_hold_expSum;
	sumBack /= (S * S); //?*?**??*?**?*?*?*?*?*?**?*?*?*?*?**

	// calc each
	for (int i = 0; i < input_width; i++)
	{
		float prevExp = *(fc_affine_hold_exp + i);
		float backVal = *(back_in_ptr + i);

		*(back_out_ptr + i) = ((backVal / S) - sumBack) * prevExp;
	
	}

}



//
void LayerClass::setup_fc_affine_kernel()
{
	// kernel pass
	this->create_kernel_util(&krn_fc_affine_pass, "fc_affine_pass");
	// arg
	this->set_kernel_arg_mem(&krn_fc_affine_pass, 0, &mem_input_data);
	this->set_kernel_arg_mem(&krn_fc_affine_pass, 1, &mem_output_data);

	// kernel batchnorm
	this->create_kernel_util(&krn_fc_affine_batchnorm, "fc_affine_batchnorm");
	// arg
	this->set_kernel_arg_mem(&krn_fc_affine_batchnorm, 0, &mem_input_data);
	this->set_kernel_arg_mem(&krn_fc_affine_batchnorm, 1, &mem_fc_affine_hold_deviation);
	this->set_kernel_arg_mem(&krn_fc_affine_batchnorm, 2, &mem_output_data);
	this->set_kernel_arg_val(&krn_fc_affine_batchnorm, 3, input_width);

	// kernel calc softmax
	this->create_kernel_util(&krn_fc_affine_softmax, "fc_affine_softmax");
	// arg
	this->set_kernel_arg_mem(&krn_fc_affine_softmax, 0, &mem_input_data);
	this->set_kernel_arg_mem(&krn_fc_affine_softmax, 1, &mem_fc_affine_hold_exp);
	this->set_kernel_arg_mem(&krn_fc_affine_softmax, 2, &mem_fc_affine_hold_expSum);
	this->set_kernel_arg_mem(&krn_fc_affine_softmax, 3, &mem_output_data);
	this->set_kernel_arg_val(&krn_fc_affine_softmax, 4, input_width);


	///////////////////////////////

	// backP kernel
	this->create_kernel_util(&krn_fc_affine_back_pass, "fc_affine_backP_pass");
	// arg
	this->set_kernel_arg_mem(&krn_fc_affine_back_pass, 0, &mem_back_in);
	this->set_kernel_arg_mem(&krn_fc_affine_back_pass, 1, &mem_back_out);

	// backP softmax kernel
	this->create_kernel_util(&krn_fc_affine_back_batchnorm, "fc_affine_backP_batchnorm");
	// arg
	this->set_kernel_arg_mem(&krn_fc_affine_back_batchnorm, 0, &mem_back_in);
	this->set_kernel_arg_mem(&krn_fc_affine_back_batchnorm, 1, &mem_output_data);
	this->set_kernel_arg_mem(&krn_fc_affine_back_batchnorm, 2, &mem_fc_affine_hold_deviation);
	this->set_kernel_arg_mem(&krn_fc_affine_back_batchnorm, 3, &mem_back_out);
	this->set_kernel_arg_mem(&krn_fc_affine_back_batchnorm, 4, &mem_fc_affine_temp_variable);
	this->set_kernel_arg_val(&krn_fc_affine_back_batchnorm, 5, input_width);

	// backP softmax kernel
	this->create_kernel_util(&krn_fc_affine_back_softmax, "fc_affine_backP_softmax");
	// arg
	this->set_kernel_arg_mem(&krn_fc_affine_back_softmax, 0, &mem_back_in);
	this->set_kernel_arg_mem(&krn_fc_affine_back_softmax, 1, &mem_fc_affine_hold_exp);
	this->set_kernel_arg_mem(&krn_fc_affine_back_softmax, 2, &mem_fc_affine_hold_expSum);
	this->set_kernel_arg_mem(&krn_fc_affine_back_softmax, 3, &mem_back_out);
	this->set_kernel_arg_val(&krn_fc_affine_back_softmax, 4, input_width);

}


void LayerClass::run_fc_affine_kernel()
{
	cl_int error;
	size_t off_2D[2] = { 0, 0 };
	size_t work_2D[2];
	size_t local_2D[2];
	switch (fc_affine_mode)
	{
	case 0:
		work_2D[0] = input_width;
		work_2D[1] = num_GPU_image;
		local_2D[0] = 2;
		local_2D[1] = 50;

		error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
			krn_fc_affine_pass,
			2,
			off_2D, work_2D, local_2D,
			0, NULL, NULL);
		if (error != CL_SUCCESS) { printf("[krn_pass] fail....%d\n", error); }
		break;

	case 1:
		work_2D[0] = 1;
		work_2D[1] = num_GPU_image;
		local_2D[0] = 1;
		local_2D[1] = 50;

		error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
			krn_fc_affine_batchnorm,
			2,
			off_2D, work_2D, local_2D,
			0, NULL, NULL);
		if (error != CL_SUCCESS) { printf("[krn_batcknorm] fail....%d\n", error); }

		break;

	case 2:
		work_2D[0] = 1;
		work_2D[1] = num_GPU_image;
		local_2D[0] = 1;
		local_2D[1] = 50;

		error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
			krn_fc_affine_softmax,
			2,
			off_2D, work_2D, local_2D,
			0, NULL, NULL);

		if (error != CL_SUCCESS) { printf("[krn_softmax] fail....%d\n", error); }
		break;
	}
}


void LayerClass::run_fc_affine_back_kernel()
{
	cl_int error;
	size_t off_2D[2] = { 0, 0 };
	size_t work_2D[2] = { 1, num_GPU_image };
	size_t local_2D[2] = { 1, 10 };

	switch (fc_affine_mode)
	{
	case 0:
		work_2D[0] = input_width;
		work_2D[1] = num_GPU_image;
		local_2D[0] = 2;
		local_2D[1] = 50;

		error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
			krn_fc_affine_back_pass,
			2,
			off_2D, work_2D, local_2D,
			0, NULL, NULL);

		if (error != CL_SUCCESS) { printf("[krn_back_pass] fail....%d\n", error); }
		break;
		
	case 1:
		work_2D[0] = 1;
		work_2D[1] = num_GPU_image;
		local_2D[0] = 1;
		local_2D[1] = 50;

		error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
			krn_fc_affine_back_batchnorm,
			2,
			off_2D, work_2D, local_2D,
			0, NULL, NULL);

		if (error != CL_SUCCESS) { printf("[krn_back_batchnorm] fail....%d\n", error); }
		break;

	case 2:
		work_2D[0] = 1;
		work_2D[1] = num_GPU_image;
		local_2D[0] = 1;
		local_2D[1] = 50;

		error = clEnqueueNDRangeKernel(cl_obj->cl_CMQ_obj[0],
			krn_fc_affine_back_softmax,
			2,
			off_2D, work_2D, local_2D,
			0, NULL, NULL);

		if (error != CL_SUCCESS) { printf("[krn_back_softmax] fail....%d\n", error); }
		break;
	}
}