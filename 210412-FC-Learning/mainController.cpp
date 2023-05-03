#include "mainController.h"


mainController::mainController()
{
	srand(time(NULL));

	printf("mainControlle init\n");

	MINST_obj = new MINST_Data_Class();
	MINST_obj->load_training_label();
	MINST_obj->load_check_label();
	MINST_obj->load_training_image();
	MINST_obj->load_check_image();

	stopwatch_obj = new stopWatch();

	// check omp thread
	int numThread = std::thread::hardware_concurrency();
	omp_set_num_threads(numThread / 2);
	//omp_set_num_threads(1);

#pragma omp parallel
	{
		ACT_OMP_THREAD = omp_get_num_threads();
		printf("*** support openMP %d %d***\n", ACT_OMP_THREAD, numThread);	}


	// layer in-out num
	int NN[10];
	NN[0] = 28 * 28;
	NN[1] = 100;
	NN[2] = 50;
	NN[3] = 10;

	// init fcl manager
	fcl_manager_obj = new FCL_manager();
	// set variables
	fcl_manager_obj->MINST_obj = MINST_obj;
	fcl_manager_obj->num_learn_image = NUM_LEARN_IMAGE;
	fcl_manager_obj->num_CPU_image = NUM_CPU_IMAGE;
	fcl_manager_obj->num_GPU_image = NUM_GPU_IMAGE;
	fcl_manager_obj->num_omp_thread = ACT_OMP_THREAD;

	// set batch image & answer set
	fcl_manager_obj->shuffule_image_answer(true, 0);

	// setup layer-chain
	fcl_manager_obj->add_fc_affine(NN[0], 0);
	fcl_manager_obj->add_fc_ReLU(NN[0], NN[1], 0); // 0-He, 1-Xavier
	fcl_manager_obj->add_fc_affine(NN[1], 0); 
	fcl_manager_obj->add_fc_ReLU(NN[1], NN[2], 0); 
	fcl_manager_obj->add_fc_affine(NN[2], 0); 
	fcl_manager_obj->add_fc_ReLU(NN[2], NN[3], 0); 
	fcl_manager_obj->add_fc_affine(NN[3], 0); // 0-pass, 1-batch norm 2- softmax
	fcl_manager_obj->add_fc_CEMS(NN[3], 1);// 0-crossE, 1-meanS

	// setup LAYER_CHAIN_CL
	fcl_manager_obj->setup_CL_layer_chain(); // setup cl_mem, kernels



	int maxCount = 0;

	for (int epoch = 0; epoch < 20; epoch++)
	{
		for (int LOOP = 0; LOOP < 10000; LOOP++)
		{
			//stopwatch_obj->start_stop_watch();

			// learn
			fcl_manager_obj->learn(LOOP);
			// modify
			fcl_manager_obj->modify();
			// clear
			fcl_manager_obj->clear();

			//stopwatch_obj->finish_stop_watch();
			//printf("time %1.2fms\n", stopwatch_obj->SECOND * 1000.0);


			if (this->check_reset_condition(LOOP))
			{
				//
				printf("\n*** reset learning [LOOP %d]***\n\n", LOOP);
				fcl_manager_obj->reset_learning();
				LOOP = 0;
			}

		} // LOOP

		// final check
		int num_rightAnswer = fcl_manager_obj->check_accuracy();

		if (maxCount < num_rightAnswer)
		{
			maxCount = num_rightAnswer;
			this->writeFile("bestAnswer", maxCount);
		}
		fcl_manager_obj->reset_learning();

	} // epoch



}

mainController::~mainController()
{

}

bool mainController::check_reset_condition(int LOOP)
{
	bool retBool = false;
	int rightImage = 0;
	// every frame image shuffled

	if (LOOP % 100 == 0)
	{
		fcl_manager_obj->shuffule_image_answer(true, 0);
	}

	if (LOOP % 300 == 0)
	{
		rightImage = fcl_manager_obj->check_accuracy();
	}

	// reset condtions
	if (fcl_manager_obj->integral < 0.8)
	{
		retBool = true;
	}

	if (LOOP == 900 && rightImage < 1000)
	{
		retBool = true;
	}
	else if (LOOP == 1800 && rightImage < 3000 )
	{
		retBool = true;
	}
	else if (LOOP == 3600 && rightImage < 5000)
	{
		retBool = true;
	}
	else if (LOOP == 6000 && rightImage < 8000 )
	{
		retBool = true;
	}

	return retBool;
}




void mainController::writeFile(const char* fileName, int NUM)
{
	std::string cppStr(fileName);
	std::string numStr(std::to_string(NUM));

	std::string savePath = cppStr + numStr;

	printf("path : %s\n", savePath.c_str());

	FILE* fp;
	errno_t err = fopen_s(&fp, savePath.c_str(), "w");
	if (err != 0)
	{
		printf("file open fail");
	}

	const char* testString = "write test";

	fwrite(testString, sizeof(char), 8, fp);

	fclose(fp);
}