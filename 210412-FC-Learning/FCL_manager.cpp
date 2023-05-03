#include "FCL_manager.h"


FCL_manager::FCL_manager()
{
	cl_manager_obj = new OpenCL_Manager_Class();
	cl_manager_obj->setup_openCL();
	cl_manager_obj->create_program_from_file(0, 0, "kernels.cl");

	stopwatch_obj = new stopWatch();
}


FCL_manager::~FCL_manager()
{}



void FCL_manager::shuffule_image_answer(bool isRandom, int startIDX)
{

	/////////////////////////////////

	// reset image & anser mem
	if (mem_GPU_image == NULL)
	{
		// create cl_mem
		cl_int error;
		long dataSize = sizeof(float) * 28 * 28 * num_GPU_image;
		float* tempMemory = (float*)malloc(dataSize);
		memset(tempMemory, 0, dataSize);

		mem_GPU_image = clCreateBuffer(cl_manager_obj->cl_CTX_obj[0],
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			dataSize,
			tempMemory,
			&error);

		if (error == CL_SUCCESS) { printf("create cl_mem(image) for GPU SUCCESS!\n"); }

		free(tempMemory);
	}

	if (mem_answer == NULL)
	{
		cl_int error;
		long dataSize = sizeof(float) * 10 * num_GPU_image;
		float* tempMemory = (float*)malloc(dataSize);
		memset(tempMemory, 0, dataSize);

		mem_answer = clCreateBuffer(cl_manager_obj->cl_CTX_obj[0],
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			dataSize,
			tempMemory,
			&error);

		if (error == CL_SUCCESS) { printf("create cl_mem(answer) for GPU SUCCESS!\n"); }

		free(tempMemory);
	}


	// decide image start index
	int start_index = 0;
	if (isRandom)
	{
		int IDX_range = 59999 - num_learn_image;
		int randVal = rand() * 2;
		// limit value
		randVal = fmin(IDX_range, randVal);

		start_index = randVal;
	}
	else
	{
		start_index = startIDX;
	}



	printf("\n- shuffle start IDX [%d] - \n\n", start_index);


	// decide CPU access ID
	for (int i = 0; i < num_CPU_image; i++)
	{
		access_ID[i] = (start_index + num_GPU_image) + i;
	}


	// update mem object contents
	//*?*?**?*?*?*?*?*?**?*?*?*?***?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?
	long dataSize_img = sizeof(float) * 28 * 28 * num_GPU_image;
	long dataSize_ans = sizeof(float) * 10 * num_GPU_image;
	//*?*?**?*?*?*?*?*?**?*?*?*?***?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?

	float* newImage_ptr = (float*)malloc(dataSize_img);
	float* newAnswer_ptr = (float*)malloc(dataSize_ans);

	float* imgHead = MINST_obj->get_train_image_float_ptr(start_index);
	float* answerHead = MINST_obj->get_train_label_float(start_index);

	memcpy(newImage_ptr, imgHead, dataSize_img);
	memcpy(newAnswer_ptr, answerHead, dataSize_ans);



	// update mem object
	clEnqueueWriteBuffer(cl_manager_obj->cl_CMQ_obj[0],
		mem_GPU_image,
		CL_FALSE,
		0,
		dataSize_img,
		newImage_ptr,
		0, NULL, NULL);

	clEnqueueWriteBuffer(cl_manager_obj->cl_CMQ_obj[0],
		mem_answer,
		CL_FALSE,
		0,
		dataSize_ans,
		newAnswer_ptr,
		0, NULL, NULL);

	clFinish(cl_manager_obj->cl_CMQ_obj[0]);

	free(newImage_ptr);
	free(newAnswer_ptr);
}


void FCL_manager::add_layer_common()
{
	// cl chain
	LAYER_CHAIN_CL[layer_count] = new LayerClass();
	LAYER_CHAIN_CL[layer_count]->num_CPU_image = num_CPU_image;
	LAYER_CHAIN_CL[layer_count]->num_GPU_image = num_GPU_image;
	LAYER_CHAIN_CL[layer_count]->answer_width = 10;
	LAYER_CHAIN_CL[layer_count]->cl_obj = cl_manager_obj;

	// OMP chain
	for (int i = 0; i < (num_omp_thread + 1); i++) // THREAD + 1
	{
		// +1 is used for accuracy test
		LAYER_CHAIN_OMP[i][layer_count] = new LayerClass();
		LAYER_CHAIN_OMP[i][layer_count]->num_CPU_image = num_CPU_image;
		LAYER_CHAIN_OMP[i][layer_count]->num_GPU_image = num_GPU_image;
		LAYER_CHAIN_OMP[i][layer_count]->cl_obj = cl_manager_obj;
	}
}


void FCL_manager::add_fc_affine(int inW, int mode)
{
	this->add_layer_common();

	// CL 
	LAYER_CHAIN_CL[layer_count]->init_as_fc_affine(inW, mode);
	// OMP
	for (int i = 0; i < num_omp_thread + 1; i++)
	{
		LAYER_CHAIN_OMP[i][layer_count]->init_as_fc_affine(inW, mode);
	}
	///////////////////////////
	layer_count++;
}

void FCL_manager::add_fc_ReLU(int inW, int outW, int iniWeightMode)
{
	this->add_layer_common();

	// CL 
	LAYER_CHAIN_CL[layer_count]->init_as_fc_ReLU(inW, outW, iniWeightMode);
	// OMP
	for (int i = 0; i < num_omp_thread + 1; i++)
	{
		LAYER_CHAIN_OMP[i][layer_count]->init_as_fc_ReLU(inW, outW, iniWeightMode);

		// copy initial weight from LAYER_CHAIN_CL
		LAYER_CHAIN_OMP[i][layer_count]->copy_weight_bias_from_layer(LAYER_CHAIN_CL[layer_count]);
	}
	/////////////////////////
	layer_count++;
}

void FCL_manager::add_fc_CEMS(int inW, int mode)
{
	this->add_layer_common();

	// CL
	LAYER_CHAIN_CL[layer_count]->init_as_fc_CEMS(inW, mode);
	// OMP
	for (int i = 0; i < num_omp_thread + 1; i++)
	{
		LAYER_CHAIN_OMP[i][layer_count]->init_as_fc_CEMS(inW, mode);
	}
	////////////////////////////
	layer_count++;
}



void FCL_manager::setup_CL_layer_chain()
{
	for (int D = 0; D < layer_count; D++)
	{
		// create each cl_mem
		LAYER_CHAIN_CL[D]->setup_cl_mem();
		// set global mem answer
		LAYER_CHAIN_CL[D]->mem_answer = mem_answer;
	}


	// setup forward mem chain
	cl_mem inMem = mem_GPU_image; // first layer's input
	for (int D = 0; D < layer_count; D++)
	{
		// set input cl_mem
		LAYER_CHAIN_CL[D]->mem_input_data = inMem;
		// update inMem
		inMem = LAYER_CHAIN_CL[D]->mem_output_data;
	}

	// setup back mem chain
	cl_mem backMem = NULL; // last layers back-in mem
	for (int D = layer_count-1; D >= 0 ; D--)
	{
		// set back-in mem
		LAYER_CHAIN_CL[D]->mem_back_in = backMem;
		// update back-in mem for next
		backMem = LAYER_CHAIN_CL[D]->mem_back_out;
	}

	for (int D = 0; D < layer_count; D++)
	{
		// setup kernels
		LAYER_CHAIN_CL[D]->setup_cl_kernel();
	}
}








void FCL_manager::learn(int LOOP)
{
	bool isCheckTimeRatio = false;

	//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	//CL
	if (isCheckTimeRatio)
	{ stopwatch_obj->start_stop_watch();}


	for (int D = 0; D < layer_count; D++)
	{
		LAYER_CHAIN_CL[D]->learn_CL();
	}

	for (int D = layer_count - 1; D >= 0; D--)
	{
		LAYER_CHAIN_CL[D]->back_propagation_CL();
	}

	for (int D = 0; D < layer_count; D++)
	{
		LAYER_CHAIN_CL[D]->readback_CL();
	}
	
	/*
	float tempPtr[784];
	cl_mem targetMem = LAYER_CHAIN_CL[0]->mem_input_data;
	clEnqueueReadBuffer(cl_manager_obj->cl_CMQ_obj[0],
		targetMem,
		CL_FALSE, 0,
		sizeof(float) * 784,
		tempPtr,
		0, NULL, NULL);
	*/

	if (isCheckTimeRatio)
	{
		//*?*?****?*?*?*?*?*?*?*?*?*?*?*?*?*?**?*?*?*?*
		clFinish(cl_manager_obj->cl_CMQ_obj[0]);
		stopwatch_obj->finish_stop_watch();
		printf("cl time %f | ", stopwatch_obj->SECOND * 1000.0);
		stopwatch_obj->start_stop_watch();
		//*?*?****?*?*?*?*?*?*?*?*?*?*?*?*?*?**?*?*?*?*
	}

	//;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	//CPU
	float* inPtr[MAX_OMP_THREAD];
	float* answerPtr[MAX_OMP_THREAD];
	float* prevBackPtr[MAX_OMP_THREAD];


#pragma omp parallel for
	for (int i = 0; i < num_CPU_image; i++)
	{
		// get omp id
		int ompid = omp_get_thread_num();

		// initial input is image data
		inPtr[ompid] = MINST_obj->get_train_image_float_ptr(access_ID[i]);
		answerPtr[ompid] = MINST_obj->get_train_label_float(access_ID[i]);

		// learn *****************************************************
		for (int D = 0; D < layer_count; D++)
		{
			LAYER_CHAIN_OMP[ompid][D]->learn(inPtr[ompid], answerPtr[ompid]);
			// update next input
			inPtr[ompid] = LAYER_CHAIN_OMP[ompid][D]->output_data_ptr;
		}

		// back propagation ::::::::::::::::::::::::::::::::::::::::::::
		prevBackPtr[ompid] = nullptr; // final layer don't need back-in ptr
		for (int B = layer_count - 1; B >= 0; B--)
		{
			LAYER_CHAIN_OMP[ompid][B]->back_propagation(prevBackPtr[ompid]);
			// update backPtr
			prevBackPtr[ompid] = LAYER_CHAIN_OMP[ompid][B]->back_out_ptr;
		}

	} // for OMP

	if (isCheckTimeRatio)
	{
		//**>?*>***??*?*?*?*?*?***?*?*?*?*?**?*/
		stopwatch_obj->finish_stop_watch();
		printf("CPU time %f\n", stopwatch_obj->SECOND * 1000.0);
		//**>?*>***??*?*?*?*?*?***?*?*?*?*?**?*/
	}

	// wait for CL completion
	clFinish(cl_manager_obj->cl_CMQ_obj[0]);


	// average loss ////////////////////////////////////////////////////
	prev_average_loss = average_loss;

	// add OMP loss
	float SUM = 0.0;
	for (int i = 0; i < num_omp_thread; i++)
	{
		float* losssum_ptr = LAYER_CHAIN_OMP[i][layer_count - 1]->fc_CEMS_result_loss_sum;
		SUM += *losssum_ptr;
	}

	// add CL loss
	SUM += *(LAYER_CHAIN_CL[layer_count - 1]->fc_CEMS_result_loss_sum);

	average_loss = SUM / (float)num_learn_image;

	////////////////////////////////////////////////////////////////////

	if (LOOP % 10 == 0)
	{
		// check CL loss val
		float* clLoss = LAYER_CHAIN_CL[layer_count - 1]->fc_CEMS_result_loss_sum;

		printf("[%d] average loss [%1.6f] : learn_coef [%1.6f] : integral [%1.5f]\n",LOOP, average_loss, learn_coef, integral);
		printf("GPU LOSS %1.4f / CPU LOSS %1.4f \n", *clLoss, SUM - *clLoss);

	}

}



void FCL_manager::modify()
{
	// sum all weight & bias sum
	for (int i = 1; i < num_omp_thread; i++)
	{
		for (int D = 0; D < layer_count; D++)
		{
			LayerClass* srcLayer = LAYER_CHAIN_OMP[i][D];
			LAYER_CHAIN_OMP[0][D]->add_weight_bias_delta_sum(srcLayer);
		}
	}

	// add CL's delta sum

	for (int D = 0; D < layer_count; D++)
	{
		LayerClass* srcLayer = LAYER_CHAIN_CL[D];
		LAYER_CHAIN_OMP[0][D]->add_weight_bias_delta_sum(srcLayer);
	}




	// average weight & bias sum
	//*********************************************
	float aveCoef = 1.0 / (float)num_learn_image; // TEMPORALY
	//*?*?*?*?*?***?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?


	for (int D = 0; D < layer_count; D++)
	{
		LAYER_CHAIN_OMP[0][D]->average_weight_bias_delta_sum( aveCoef );
	}


	// update weight
	if (average_loss > 1.0) // loss reduced
	{learn_coef = 1.0;}
	else
	{learn_coef = pow(average_loss, 1.0);} // meanS = ave^1, crossE = ave^3

	if (average_loss < prev_average_loss)
	{integral *= 1.0005;}
	else
	{integral *= 0.999;}

	if (integral > 2.0)
	{integral = 2.0;}

	for (int D = 0; D < layer_count; D++)
	{
		//LAYER_CHAIN_OMP[0][D]->modify_weight_bias_by_SDG(0.05*learn_coef*integral);
		LAYER_CHAIN_OMP[0][D]->modify_weight_bias_by_Momentom(0.01*learn_coef*integral, 0.85);
	}


	// copy weight
	for (int i = 1; i < num_omp_thread; i++)
	{
		for (int D = 0; D < layer_count; D++)
		{
			LAYER_CHAIN_OMP[i][D]->copy_weight_bias_from_layer(LAYER_CHAIN_OMP[0][D]);
		}
	}

	//*?*?*?*?***?*?*?*?*?*?*?*?*?*?*?*/
	// copy weight to CL mem
	//*?*?*?*?***?*?*?*?*?*?*?*?*?*?*?*/
	for (int D = 0; D < layer_count; D++)
	{
		LAYER_CHAIN_CL[D]->update_cl_mem_weight(LAYER_CHAIN_OMP[0][D]);
	}

}


void FCL_manager::clear()
{
	// clear ReLU layer's weight & bias delta sum
	for (int i = 0; i < num_omp_thread; i++)
	{
		for (int D = 0; D < layer_count; D++)
		{
			LAYER_CHAIN_OMP[i][D]->clear_weight_bias_delta_sum();
		}

		// clear loss sum
		*(LAYER_CHAIN_OMP[i][layer_count - 1]->fc_CEMS_result_loss_sum) = 0.0;
	}

	//*>***?*?*?**?***?*?*?*?*?*?*?*?*?
	// CL mem's weight-sum & bias-sum & lossSum is overwrote, everytime. 
	//*>***?*?*?**?***?*?*?*?*?*?*?*?*?

}





void FCL_manager::reset_learning()
{
	// change learning image set
	this->shuffule_image_answer(true, 0);


	// update layers weight & bias
	for (int D = 0; D < layer_count; D++)
	{
		LAYER_CHAIN_OMP[0][D]->generate_initial_HX_weight(0);
	}

	for (int i = 1 ; i < num_omp_thread; i++)
	{
		for (int D = 0; D < layer_count; D++)
		{
			LAYER_CHAIN_OMP[i][D]->copy_weight_bias_from_layer(LAYER_CHAIN_OMP[0][D]);
		}
	}
	

	// clear weight&bias delta sum
	this->clear();

	// reset weight & bias velocity
	
	for (int i = 1; i < num_omp_thread; i++)
	{
		for (int D = 0; D < layer_count; D++)
		{
			LAYER_CHAIN_OMP[i][D]->clear_weight_bias_velocity();
		}
	}
	
	// reset variables
	integral = 1.0;
}



int FCL_manager::check_accuracy()
{
	int COUNT = 0;

	for (int i = 0; i < 10000; i++)
	{
		float* imgPtr = MINST_obj->get_check_image_float_ptr(i);
		float* ansPtr = MINST_obj->get_check_label_float(i);

		float* inPtr = imgPtr;
		for (int D = 0; D < layer_count; D++)
		{
			LAYER_CHAIN_OMP[0][D]->learn(inPtr, ansPtr);
			inPtr = LAYER_CHAIN_OMP[0][D]->output_data_ptr;
		}

		// get anser
		float* resultPtr = LAYER_CHAIN_OMP[0][layer_count - 2]->output_data_ptr;

		float maxVal = 0.0;
		int maxIDX = 0;
		for (int a = 0; a < 10; a++)
		{
			if (*(resultPtr + a) > maxVal)
			{
				maxVal = *(resultPtr + a);
				maxIDX = a;
			}
		}
	
		int ansIDX = MINST_obj->get_check_label(i);

		if (ansIDX == maxIDX)
		{
			COUNT++;
		}
	}

	printf("Accuracy %d (%f%%)\n", COUNT, (float)COUNT / 100.0);
	return COUNT;
}