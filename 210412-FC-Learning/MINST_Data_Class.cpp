#include "MINST_Data_Class.h"

MINST_Data_Class::MINST_Data_Class()
{
	printf("\n*** MINST init ***\n");

}


MINST_Data_Class::~MINST_Data_Class()
{
	
}



int MINST_Data_Class::byteSwap_4(int num)
{
	int four[4];
	int retVal = 0;

	four[0] = (num & 0xFF000000) >> 24;
	four[1] = (num & 0x00FF0000) >> 8;
	four[2] = (num & 0x0000FF00) << 8;
	four[3] = (num & 0x000000FF) << 24;
	
	printf("%x : %x : %x : %x\n", four[0], four[1], four[2], four[3]);

	retVal = retVal | four[3];
	retVal = retVal | four[2];
	retVal = retVal | four[1];
	retVal = retVal | four[0];

	return retVal;
}


void MINST_Data_Class::load_training_label()
{
	FILE* fp;
	errno_t err = fopen_s(&fp, "MINDATA/train-labels.idx1-ubyte", "rb");
	if (err != 0)
	{
		printf("open training label fail... return\n");
		return;
	}

	printf("\n*** LOAD TRAINING LABEL [%d] 0 is success***\n", err);

	//////////////////////////////////////////////////////////////////////

	// get data
	int identifier;
	int numLabel;
	int numRead;

	// get identifier & num of label
	numRead = fread_s(&identifier, sizeof(int), 4, 1, fp);
	identifier = this->byteSwap_4(identifier);
	printf("fread_s[%d] identifier : %d\n",numRead, identifier);

	numRead = fread_s(&numLabel, sizeof(int), 4, 1, fp);
	numLabel = this->byteSwap_4(numLabel);
	printf("fread_s[%d] num label : %d\n", numRead, numLabel);



	for (int i = 0; i < numLabel; i++) // 60000
	{
		fread_s(&train_label[i], sizeof(unsigned char), 1, 1, fp);

		for (int n = 0; n < 10; n++)
		{
			if (n == train_label[i])
			{
				train_label_float[i][n] = 1.0;
			}
			else
			{
				train_label_float[i][n] = 0.0;
			}
		}
	}
}



void MINST_Data_Class::load_check_label()
{
	FILE* fp;

	errno_t err = fopen_s(&fp, "MINDATA/t10k-labels.idx1-ubyte", "rb");

	if (err != 0)
	{
		printf("open check data fail... return\n");
		return;
	}

	printf("\n*** LOAD CHECK LABEL [%d] 0 is success***\n", err);


	// get identifier & num check label
	int identifier;
	fread_s(&identifier, sizeof(int), 4, 1, fp);
	identifier = this->byteSwap_4(identifier);
	printf("check label identifier : %x\n", identifier);

	int numLabel;
	fread_s(&numLabel, sizeof(int), 4, 1, fp);
	numLabel = this->byteSwap_4(numLabel);
	printf("num check label : %d\n", numLabel);



	for (int i = 0; i < numLabel; i++)// 10000
	{
		fread_s(&check_label[i], sizeof(unsigned char), 1, 1, fp);

		for (int n = 0; n < 10; n++)
		{
			if (n == check_label[i])
			{
				check_label_float[i][n] = 1.0;
			}
			else
			{
				check_label_float[i][n] = 0.0;
			}
		}
	}
}



void MINST_Data_Class::load_training_image()
{
	FILE* fp;

	errno_t err = fopen_s(&fp, "MINDATA/train-images.idx3-ubyte", "rb");
	if (err != 0)
	{
		printf("load training image fail... return\n");
		return;
	}

	printf("\n*** LOAD TRAINING IMAGE [%d]\n", err);

	// check header
	int identifier, numImage, width, height;
	fread_s(&identifier, sizeof(int), 4, 1, fp);
	fread_s(&numImage, sizeof(int), 4, 1, fp);
	fread_s(&width, sizeof(int), 4, 1, fp);
	fread_s(&height, sizeof(int), 4, 1, fp);

	identifier = this->byteSwap_4(identifier);
	numImage = this->byteSwap_4(numImage);
	width = this->byteSwap_4(width);
	height = this->byteSwap_4(height);

	printf("identifier : %x\n", identifier);
	printf("num image : %d\n", numImage);
	printf("width : %d\n", width);
	printf("height : %d\n", height);

	for (int i = 0; i < 60000; i++)
	{
		for (int y = 0; y < 28; y++)
		{
			for (int x = 0; x < 28; x++)
			{
				fread_s(&train_image[i][y][x],
							sizeof(unsigned char),
							1,
							1,
							fp);

				// normal
				train_image_float[i][y][x] = (float)(train_image[i][y][x]) / 255.0;

				// 3ch
				train_image_3ch_float[i][y][x][0] = train_image_float[i][y][x];
				train_image_3ch_float[i][y][x][1] = train_image_float[i][y][x];
				train_image_3ch_float[i][y][x][2] = train_image_float[i][y][x];
			}
		}
	}
}




void MINST_Data_Class::load_check_image()
{
	FILE* fp;

	errno_t err = fopen_s(&fp, "MINDATA/t10k-images.idx3-ubyte", "rb");
	if (err != 0)
	{
		printf("load check image fail.. return\n");
		return;
	}

	printf("\n*** LOAD CHECK IMAGE SUCCESS[%d]***\n", err);


	// get header
	int identifier, numImage, width, height;
	fread_s(&identifier, 4, 4, 1, fp);
	fread_s(&numImage, 4, 4, 1, fp);
	fread_s(&width, 4, 4, 1, fp);
	fread_s(&height, 4, 4, 1, fp);

	identifier = this->byteSwap_4(identifier);
	numImage = this->byteSwap_4(numImage);
	width = this->byteSwap_4(width);
	height = this->byteSwap_4(height);

	printf("identifier : %x\n", identifier);
	printf("num image : %d\n", numImage);
	printf("width : %d\n", width);
	printf("height : %d\n", height);


	// get data
	for (int i = 0; i < 10000; i++)
	{
		for (int y = 0; y < 28; y++)
		{
			for (int x = 0; x < 28; x++)
			{
				fread_s(&check_image[i][y][x], 1, 1, 1, fp);

				// normal
				check_image_float[i][y][x] = (float)(check_image[i][y][x]) / 255.0;

				// 3ch
				check_image_3ch_float[i][y][x][0] = check_image_float[i][y][x];
				check_image_3ch_float[i][y][x][1] = check_image_float[i][y][x];
				check_image_3ch_float[i][y][x][2] = check_image_float[i][y][x];
			}
		}
	}
}



unsigned char MINST_Data_Class::get_train_label(int num)
{
	return train_label[num];
}

unsigned char MINST_Data_Class::get_check_label(int num)
{
	return check_label[num];
}


float* MINST_Data_Class::get_train_label_float(int num)
{
	return &train_label_float[num][0];
}

float* MINST_Data_Class::get_check_label_float(int num)
{
	return &check_label_float[num][0];
}




unsigned char* MINST_Data_Class::get_train_image_ptr(int num)
{
	unsigned char* retPtr = &train_image[num][0][0];
	return retPtr;
}

unsigned char* MINST_Data_Class::get_check_image_ptr(int num)
{
	unsigned char* retPtr = &check_image[num][0][0];
	return retPtr;
}


float* MINST_Data_Class::get_train_image_float_ptr(int num)
{
	float* retPtr = &train_image_float[num][0][0];
	return retPtr;
}


float* MINST_Data_Class::get_check_image_float_ptr(int num)
{
	float* retPtr = &check_image_float[num][0][0];
	return retPtr;
}

float* MINST_Data_Class::get_train_image_3ch_float_ptr(int num)
{
	float* retPtr = &train_image_3ch_float[num][0][0][0];
	return retPtr;
}
float* MINST_Data_Class::get_check_image_3ch_float_ptr(int num)
{
	float* retPtr = &check_image_3ch_float[num][0][0][0];
	return retPtr;
}
