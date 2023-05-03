#include "framework.h"

#include "MINST_Data_Class.h"
#include "FCL_manager.h"
#include "stopWatch.h"

#include <omp.h>
#include <thread>
#include <string>



#define NUM_LEARN_IMAGE 3000
#define NUM_CPU_IMAGE 1300
#define NUM_GPU_IMAGE 1700

#pragma once
class mainController
{
public:

	MINST_Data_Class* MINST_obj;

	FCL_manager* fcl_manager_obj;
	stopWatch* stopwatch_obj;
	int ACT_OMP_THREAD = 1;

	mainController();
	~mainController();

	bool check_reset_condition(int LOOP);
	void writeFile(const char* fileName, int NUM);
};

