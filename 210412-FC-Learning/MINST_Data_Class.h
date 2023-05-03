#pragma once

#include "framework.h"

class MINST_Data_Class
{
public:


	MINST_Data_Class();
	~MINST_Data_Class();

	unsigned char train_label[60000];
	unsigned char check_label[10000];
	float train_label_float[60000][10];
	float check_label_float[10000][10];
	unsigned char train_image[60000][28][28];
	unsigned char check_image[10000][28][28];
	float train_image_float[60000][28][28];
	float check_image_float[10000][28][28];
	float train_image_3ch_float[60000][28][28][3];
	float check_image_3ch_float[60000][28][28][3];


	void load_check_image();
	void load_check_label();
	void load_training_image();
	void load_training_label();

	int byteSwap_4(int num);

	// return label
	unsigned char get_train_label(int num);
	unsigned char get_check_label(int num);
	float* get_train_label_float(int num);
	float* get_check_label_float(int num);

	// return image data
	unsigned char* get_train_image_ptr(int num);
	unsigned char* get_check_image_ptr(int num);
	float* get_train_image_float_ptr(int num);
	float* get_check_image_float_ptr(int num);
	float* get_train_image_3ch_float_ptr(int num);
	float* get_check_image_3ch_float_ptr(int num);
};

