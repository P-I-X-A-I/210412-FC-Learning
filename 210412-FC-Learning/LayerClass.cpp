#include "LayerClass.h"

LayerClass::LayerClass()
{

}

LayerClass::~LayerClass()
{

}

//////////////////////////////////////////////////////////////////

void LayerClass::learn(float* inPtr, float* ansPtr)
{
	if (strcmp("fcAff", layer_type_str) == 0)
	{
		this->input_data_to_fc_affine(inPtr);
	}
	else if (strcmp("fcReLU", layer_type_str) == 0)
	{
		this->input_data_to_fc_ReLU(inPtr);
	}
	else if (strcmp("fcCEMS", layer_type_str) == 0)
	{
		this->input_data_to_fc_CEMS(inPtr, ansPtr);
	}
}


void LayerClass::back_propagation(float* backInPtr)
{
	if (strcmp("fcAff", layer_type_str) == 0)
	{
		this->back_propagation_fc_affine(backInPtr);
	}
	else if (strcmp("fcReLU", layer_type_str) == 0)
	{
		this->back_propagation_fc_ReLU(backInPtr);
	}
	else if (strcmp("fcCEMS", layer_type_str) == 0)
	{
		// back-propagation is already calculated
	}
}


void LayerClass::learn_CL()
{
	if (strcmp("fcAff", layer_type_str) == 0)
	{
		this->run_fc_affine_kernel();
	}
	else if (strcmp("fcReLU", layer_type_str) == 0)
	{
		this->run_fc_ReLU_kernel();
	}
	else if (strcmp("fcCEMS", layer_type_str) == 0)
	{
		this->run_fc_CEMS_kernel();
	}
}



void LayerClass::back_propagation_CL()
{
	if (strcmp("fcAff", layer_type_str) == 0)
	{
		this->run_fc_affine_back_kernel();
	}
	else if (strcmp("fcReLU", layer_type_str) == 0)
	{
		this->run_fc_ReLU_back_kernel();
	}
	else if (strcmp("fcCEMS", layer_type_str) == 0)
	{
		this->run_fc_CEMS_back_kernel();
	}
}


void LayerClass::readback_CL()
{
	if (strcmp("fcAff", layer_type_str) == 0)
	{
		// nothing
	}
	else if (strcmp("fcReLU", layer_type_str) == 0)
	{
		this->readback_ReLU();
	}
	else if (strcmp("fcCEMS", layer_type_str) == 0)
	{
		this->readback_CEMS();
	}
}
////////////////////////////////////////////////////////////////



// setup CL layers
void LayerClass::setup_cl_mem()
{
	if (strcmp("fcAff", layer_type_str) == 0)
	{
		this->alloc_fc_affine_cl_mem();
	}
	else if (strcmp("fcReLU", layer_type_str) == 0)
	{
		this->alloc_fc_ReLU_cl_mem();
	}
	else if (strcmp("fcCEMS", layer_type_str) == 0)
	{
		this->alloc_fc_CEMS_cl_mem();
	}
}



cl_int LayerClass::create_mem_util(cl_mem* memPtr, long dSize, cl_int flag)
{
	cl_int error;
	float* tempMemory = (float*)malloc(dSize);
	memset(tempMemory, 0, dSize);

	*memPtr = clCreateBuffer(cl_obj->cl_CTX_obj[0],
		flag | CL_MEM_COPY_HOST_PTR,
		dSize,
		tempMemory,
		&error);

	free(tempMemory);

	return error;
}



void LayerClass::setup_cl_kernel()
{
	if (strcmp("fcAff", layer_type_str) == 0)
	{
		this->setup_fc_affine_kernel();
	}
	else if (strcmp("fcReLU", layer_type_str) == 0)
	{
		this->setup_fc_ReLU_kernel();
	}
	else if (strcmp("fcCEMS", layer_type_str) == 0)
	{
		this->setup_fc_CEMS_kernel();
	}
}


void LayerClass::create_kernel_util(cl_kernel* krnPtr, const char* funcName)
{
	cl_int error;

	*krnPtr = clCreateKernel(cl_obj->cl_PRG_obj[0], funcName, &error);

	if (error == CL_SUCCESS) { printf("create kernel [%s]\n", funcName); }
	
}


void LayerClass::set_kernel_arg_mem(cl_kernel* krnPtr, int argIDX, cl_mem* memPtr)
{
	cl_int error;

	error = clSetKernelArg(*krnPtr, argIDX, sizeof(cl_mem), memPtr);

	if (error == CL_SUCCESS) { printf("%p set kernel mem arg[%d]\n", krnPtr, argIDX); }
}
void LayerClass::set_kernel_arg_val(cl_kernel* krnPtr, int argIDX, cl_int val)
{
	cl_int error;
	cl_int value = val;

	error = clSetKernelArg(*krnPtr, argIDX, sizeof(cl_int), &value);
	if (error == CL_SUCCESS) { printf("%p set kernel val arg[%d]\n", krnPtr, argIDX); }
}








/// modify
void LayerClass::add_weight_bias_delta_sum(LayerClass* srcLayer)
{
	if (strcmp("fcReLU", layer_type_str) == 0)
	{
		float* weiDel_ptr = fc_ReLU_weight_delta_sum;
		float* biasDel_ptr = fc_ReLU_bias_delta_sum;

		float* srcWei_ptr = srcLayer->fc_ReLU_weight_delta_sum;
		float* srcBias_ptr = srcLayer->fc_ReLU_bias_delta_sum;

		// add bias delta sum
		for (int i = 0; i < output_width; i++)
		{
			*(biasDel_ptr + i) += *(srcBias_ptr + i);
		}

		// add weight delta sum
		for (int i = 0; i < (input_width * output_width); i++)
		{
			*(weiDel_ptr + i) += *(srcWei_ptr + i);
		}
	}
}

void LayerClass::average_weight_bias_delta_sum(float coef)
{
	if (strcmp("fcReLU", layer_type_str) == 0)
	{
		// average bias delta sum
		for (int i = 0; i < output_width; i++)
		{
			*(fc_ReLU_bias_delta_sum + i) *= coef;
		}

		// average weight delta sum
		for (int i = 0; i < (input_width * output_width); i++)
		{
			*(fc_ReLU_weight_delta_sum + i) *= coef;
		}
	}
}


void LayerClass::modify_weight_bias_by_SDG(float coef)
{
	if (strcmp("fcReLU", layer_type_str) == 0)
	{
		float* weight_ptr = fc_ReLU_weight_ptr;
		float* bias_ptr = fc_ReLU_bias_ptr;

		// update bias
		for (int i = 0; i < output_width; i++)
		{
			*(bias_ptr + i) -= (*(fc_ReLU_bias_delta_sum + i)) * coef;
		}

		// update weight
		for (int i = 0; i < (input_width * output_width); i++)
		{
			*(weight_ptr + i) -= (*(fc_ReLU_weight_delta_sum + i)) * coef;
		}
	}
}


void LayerClass::modify_weight_bias_by_Momentom(float coef, float brake)
{
	if (strcmp("fcReLU", layer_type_str) == 0)
	{
		// bias
		for (int i = 0; i < output_width; i++)
		{
			// brake velocity
			*(fc_ReLU_bias_velocity + i) *= brake;

			// add F to velocity
			*(fc_ReLU_bias_velocity + i) -= (*(fc_ReLU_bias_delta_sum + i))*coef;

			// modify weight by velocity
			*(fc_ReLU_bias_ptr + i) += *(fc_ReLU_bias_velocity + i);
		}


		for (int i = 0; i < input_width*output_width; i++)
		{
			// brake velocity
			*(fc_ReLU_weight_velocity + i) *= brake;

			// add F to velocity
			*(fc_ReLU_weight_velocity + i) -= (*(fc_ReLU_weight_delta_sum + i))*coef;

			// modify weight by velocity
			*(fc_ReLU_weight_ptr + i) += *(fc_ReLU_weight_velocity + i);
		}
	}
}





void LayerClass::copy_weight_bias_from_layer(LayerClass* srcLayer)
{
	if (strcmp("fcReLU", layer_type_str) == 0)
	{
		float* srcBiasPtr = srcLayer->fc_ReLU_bias_ptr;
		float* srcWeightPtr = srcLayer->fc_ReLU_weight_ptr;

		// copy bias
		long dataSize = output_width * sizeof(float);
		memcpy(fc_ReLU_bias_ptr, srcBiasPtr, dataSize);

		// copy weight
		dataSize = input_width * output_width * sizeof(float);		
		memcpy(fc_ReLU_weight_ptr, srcWeightPtr, dataSize);
	}
}

/// clear
void LayerClass::clear_weight_bias_delta_sum()
{
	if (strcmp("fcReLU", layer_type_str) == 0)
	{

		// clear bias delta sum
		memset(fc_ReLU_bias_delta_sum, 0, sizeof(float)*output_width);

		// clear weight delta sum
		memset(fc_ReLU_weight_delta_sum, 0, sizeof(float) * input_width * output_width );
	}
}


void LayerClass::clear_weight_bias_velocity()
{
	if (strcmp("fcReLU", layer_type_str) == 0)
	{
		// clear bias velocity
		for (int i = 0; i < output_width; i++)
		{
			*(fc_ReLU_bias_velocity + i) = 0.0;
		}
		// clear weight velocity
		for (int i = 0; i < output_width*input_width ; i++)
		{
			*(fc_ReLU_weight_velocity + i) = 0.0;
		}
	}
}


void LayerClass::update_cl_mem_weight(LayerClass* srcLayer)
{
	if (strcmp("fcReLU", layer_type_str) == 0)
	{
		cl_int error;

		// update bias & weight cl_mem
		float* src_bias = srcLayer->fc_ReLU_bias_ptr;
		float* src_weight = srcLayer->fc_ReLU_weight_ptr;

		cl_mem mem_target_bias = this->mem_fc_ReLU_bias;
		cl_mem mem_target_weight = this->mem_fc_ReLU_weight;


		// write bias 
		error = clEnqueueWriteBuffer(cl_obj->cl_CMQ_obj[0],
			mem_target_bias,
			CL_TRUE,
			0, sizeof(float)*output_width,
			src_bias,
			0, NULL, NULL);

		if (error != CL_SUCCESS) { printf("enqueue write buf fail"); }

		// write weight
		error = clEnqueueWriteBuffer(cl_obj->cl_CMQ_obj[0],
			mem_target_weight,
			CL_TRUE,
			0, sizeof(float)*output_width*input_width,
			src_weight,
			0, NULL, NULL);


		if (error != CL_SUCCESS) { printf("enqueue write buf fail"); }

		clFinish(cl_obj->cl_CMQ_obj[0]);
	}
}