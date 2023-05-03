#include "framework.h"
#include "OpenCL_Manager_Class.h"

#include <random>

#pragma once
class LayerClass
{
public:
	// COMMON //////////
	char layer_type_str[32]; // "fcAff" "fcReLU" "fcCEMS"

	bool isLog = true;

	// data variavle
	int answer_width;
	int input_width;
	int output_width;
	float* input_data_ptr = nullptr; // set by prev layer
	float* output_data_ptr = nullptr;
	float* back_in_ptr = nullptr; // set by prev layer
	float* back_out_ptr = nullptr;
	int num_CPU_image = 0;
	int num_GPU_image = 0;

	cl_mem mem_input_data;
	cl_mem mem_output_data;
	cl_mem mem_back_in;
	cl_mem mem_back_out;
	cl_mem mem_answer; // used only in CEMS

	// pointer for read-back
	//float* cl_bias_delta_sum_ptr = nullptr; // outW
	//float* cl_weight_delta_sum_ptr = nullptr; // inW * outW
	//float* cl_result_loss_ptr = nullptr; // num_Image


	// openCL
	OpenCL_Manager_Class* cl_obj;

	LayerClass();
	~LayerClass();

	// setup
	void init_as_fc_affine(int inW, int mode);
	void init_as_fc_ReLU(int inW, int outW, int iniWeightMode);
	void init_as_fc_CEMS(int inW, int mode);
	
	// setup ( CL )
	void setup_cl_mem();
	cl_int create_mem_util(cl_mem* memPtr, long dSize, cl_int flag);
	void setup_cl_kernel();
	void create_kernel_util(cl_kernel* krnPtr, const char* funcName);
	void set_kernel_arg_mem(cl_kernel* krnPtr, int argIDX, cl_mem* memPtr);
	void set_kernel_arg_val(cl_kernel* krnPtr, int argIDX, cl_int val);


	// learn
	void learn(float* inPtr, float* ansPtr);
	void back_propagation(float* backInPtr);

	// learn with CL
	void learn_CL();
	void back_propagation_CL();
	void readback_CL();
	void update_cl_mem_weight(LayerClass* srcLayer);

	// modify
	void add_weight_bias_delta_sum(LayerClass* srcLayer);
	void average_weight_bias_delta_sum(float coef);
	void modify_weight_bias_by_SDG(float coef);
	void modify_weight_bias_by_Momentom(float coef, float brake);

	void copy_weight_bias_from_layer(LayerClass* srcLayer);



	// clear
	void clear_weight_bias_delta_sum();
	void clear_weight_bias_velocity();




	///////// Affine /////////////////////
	int fc_affine_mode = 0; // 0-pass, 1-batchnorm, 2- softmax

	float* fc_affine_hold_deviation;
	float* fc_affine_hold_exp;
	float* fc_affine_hold_expSum;
	cl_mem mem_fc_affine_hold_deviation;
	cl_mem mem_fc_affine_hold_exp;
	cl_mem mem_fc_affine_hold_expSum;
	cl_mem mem_fc_affine_temp_variable; // for calculation of backP of batch norm

	cl_kernel krn_fc_affine_pass;
	cl_kernel krn_fc_affine_batchnorm;
	cl_kernel krn_fc_affine_softmax;

	cl_kernel krn_fc_affine_back_pass;
	cl_kernel krn_fc_affine_back_batchnorm;
	cl_kernel krn_fc_affine_back_softmax;



	void alloc_fc_affine_CPU_memory();
	void input_data_to_fc_affine(float* inPtr);
	void back_propagation_fc_affine(float* backInPtr);

	// affine CL
	void alloc_fc_affine_cl_mem();
	void setup_fc_affine_kernel();

	void run_fc_affine_kernel();
	void run_fc_affine_back_kernel();

private:
	void batch_normalization();
	void softmax();
	void back_batch_normalization();
	void back_softmax();


	//////// ReLU ////////////////////////
public:
	float* fc_ReLU_weight_ptr = nullptr; // inW * outW
	float* fc_ReLU_bias_ptr = nullptr; // outW
	cl_mem mem_fc_ReLU_weight; // inW * outW
	cl_mem mem_fc_ReLU_bias; // outW

	float* fc_ReLU_bias_delta_sum; // outW
	float* fc_ReLU_weight_delta_sum; // inW * outW
	float* fc_ReLU_bias_velocity; // outW
	float* fc_ReLU_weight_velocity; // inW * outW
	cl_mem mem_fc_ReLU_bias_delta_sum;
	cl_mem mem_fc_ReLU_weight_delta_sum;
	cl_mem mem_fc_ReLU_bias_delta;
	cl_mem mem_fc_ReLU_weight_delta;
	//cl_mem mem_fc_ReLU_bias_velocity;
	//cl_mem mem_fc_ReLU_weight_velocity;

	cl_kernel krn_fc_ReLU;
	cl_kernel krn_fc_ReLU_back_biasDelta;
	cl_kernel krn_fc_ReLU_sum_biasDelta;
	cl_kernel krn_fc_ReLU_back_weightDelta;
	cl_kernel krn_fc_ReLU_sum_weightDelta;
	cl_kernel krn_fc_ReLU_back_out;

	void alloc_fc_ReLU_CPU_memory(int iniMode);
	void input_data_to_fc_ReLU(float* inPtr);
	void back_propagation_fc_ReLU(float* backInPtr);
	void readback_ReLU();

	// cl
	void alloc_fc_ReLU_cl_mem();
	void setup_fc_ReLU_kernel();
	void run_fc_ReLU_kernel();
	void run_fc_ReLU_back_kernel();


	//**
	void generate_initial_HX_weight(int iniMode);
	//**




	////////// CEMS /////////////////////
	int fc_CEMS_mode = 0;
	float* fc_CEMS_result_loss_sum;


	void alloc_fc_CEMS_CPU_memory();
	void input_data_to_fc_CEMS(float* inPtr, float* ansPtr);

	cl_mem mem_fc_CEMS_loss_sum;
	cl_kernel krn_fc_crossE;
	cl_kernel krn_fc_meanS;
	cl_kernel krn_fc_CEMS_sum_loss;

	// cl
	void alloc_fc_CEMS_cl_mem();
	void setup_fc_CEMS_kernel();
	void run_fc_CEMS_kernel();
	void run_fc_CEMS_back_kernel();
	void readback_CEMS();

};

