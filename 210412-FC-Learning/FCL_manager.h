#include "framework.h"

#include "MINST_Data_Class.h"
#include "OpenCL_Manager_Class.h"
#include "stopWatch.h"

#include "LayerClass.h"

#include <omp.h>

#define MAX_OMP_THREAD 128
#define MAX_DEEPNESS 128

#pragma once
class FCL_manager
{
public:

	MINST_Data_Class* MINST_obj; // set from maincontroller
	OpenCL_Manager_Class* cl_manager_obj;
	stopWatch* stopwatch_obj;

	// learning image data set
	int num_learn_image = 0;
	int num_omp_thread = 1;
	int num_CPU_image = 0;
	int num_GPU_image = 0;
	int access_ID[60000]; // shuffle image
	cl_mem mem_GPU_image = NULL;
	cl_mem mem_answer = NULL;


	// layer chain
	int layer_count = 0;
	LayerClass* LAYER_CHAIN_CL[MAX_DEEPNESS];
	LayerClass* LAYER_CHAIN_OMP[MAX_OMP_THREAD][MAX_DEEPNESS];
	
	// learning coef
	float average_loss = 0.0;
	float prev_average_loss = 0.0;
	float learn_coef = 1.0;
	float integral = 1.0;

	FCL_manager();
	~FCL_manager();

	//
	void shuffule_image_answer(bool isRandom, int startIDX);
	void add_layer_common();
	void add_fc_affine(int inW, int mode);
	void add_fc_ReLU(int inW, int outW, int iniWeightMode);
	void add_fc_CEMS(int inW, int mode);
	void setup_CL_layer_chain();

	// learning
	void learn(int LOOP);
	void modify();
	void clear();
	//////////
	void reset_learning();
	int check_accuracy();
};

