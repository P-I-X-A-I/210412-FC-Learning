// TODO: Add OpenCL kernel code here.


__kernel void fc_affine_pass( __global float* inData,
							__global float* outData)
{

	int xID = get_global_id(0); // in-out W
	int imgID = get_global_id(1); // image

	int x_Sz = get_global_size(0); // in width

	///////////////////////////////
	int aID = (imgID * x_Sz) + xID;

	outData[aID] = inData[aID];
	///////////////////////////////
}



///////// batch norm //////////////
__kernel void fc_affine_batchnorm(
		__global float* inData,
		__global float* hold_DEVI,
		__global float* outData,
		int W
)
{
	int imgID = get_global_id(1);
	int baseID = imgID * W;

	// average
	float AVE = 0.0;
	for (int i = 0; i < W; i++)
	{
		AVE += inData[baseID + i];
	}

	AVE /= (float)W;

	// deviation
	float DEVI = 0.0;
	float tempVal = 0.0;
	for (int i = 0; i < W; i++)
	{
		tempVal = inData[baseID + i] - AVE;
		DEVI += (tempVal * tempVal);
	}

	DEVI /= (float)W;

	// hold deviation for back-propagation
	hold_DEVI[imgID] = DEVI;

	// out Data
	for (int i = 0; i < W; i++)
	{
		outData[baseID + i] = (inData[baseID + i] - AVE) / sqrt(DEVI + 0.0000001);
	}
}


__kernel void fc_affine_softmax(
	__global float* inData,
	__global float* hold_exp,
	__global float* hold_expSum,
	__global float* outData,
	int W)
{
	int imgID = get_global_id(1);

	int baseID = W * imgID;

	// find max
	float MAX = -100.0;
	for (int i = 0; i < W; i++)
	{
		MAX = fmax(MAX, inData[baseID + i]);
	}

	// exp sum
	float expVal = 0.0;
	float expSum = 0.0;
	for (int i = 0; i < W; i++)
	{
		expVal = exp(inData[baseID + i] - MAX);
		expSum += expVal;

		// hold each expVal for back-propagation
		hold_exp[baseID + i] = expVal;
	}

	// hold expSum for back-propagation
	hold_expSum[imgID] = expSum;


	// write outdata
	for (int i = 0; i < W; i++)
	{
		outData[baseID + i] = exp(inData[baseID + i] - MAX) / expSum;
	}
}




//////////////////////////////////////////////////////////////
////////////////// fc ReLU //////////////////////////////////////


__kernel void fc_ReLU(
	__global float* inData,
	__global float* weightData,
	__global float* biasData,
	__global float* outData,
	int inW // input width
)
{
	int xID = get_global_id(0); // out W
	int imgID = get_global_id(1);
	int outW = get_global_size(0);

	int inBaseID = inW * imgID;
	int weiBaseID = inW * xID; // no image layer
	int outBaseID = outW * imgID;

	// sum weight * inData
	float calSum = 0.0;
	float inVal = 0.0;
	for (int i = 0; i < inW; i++)
	{
		inVal = inData[inBaseID + i];
		calSum += weightData[weiBaseID + i] * inVal;
	}

	// add bias
	calSum += biasData[xID]; // no image layer

	// out data
	float FINAL = fmax((float)0.0, calSum);
	outData[outBaseID + xID] = FINAL;

}

/////////////////////////////////////////////////////////////
/////////////////// for corssE_meanS  

__kernel void fc_crossE(
	__global float* inData,
	__global float* answerData,
	__global float* outData,
	int ansW,
	__global float* backDelta
)
{
	int imgID = get_global_id(1);

	int BaseID = imgID * ansW;

	float SUM = 0.0;

	for (int i = 0; i < ansW; i++)
	{
		float rVal = inData[BaseID + i];
		float aVal = answerData[BaseID + i];
		float entropy = -(aVal * log(rVal + 0.0000001));

		SUM += entropy;

		// back-propagation
		backDelta[BaseID + i] = (-aVal) / (rVal + 0.0000001);
	}

	outData[imgID] = SUM; // loss value

}


__kernel void fc_meanS(
	__global float* inData,
	__global float* answerData,
	__global float* outData,
	int ansW,
	__global float* backDelta
)
{
	int imgID = get_global_id(1);

	int BaseID = imgID * ansW;

	float SUM = 0.0;

	for (int i = 0; i < ansW; i++)
	{
		float rVal = inData[BaseID + i];
		float aVal = answerData[BaseID + i];
		float meanS = (rVal - aVal)*(rVal - aVal);

		SUM += meanS;

		// back-propagation
		backDelta[BaseID + i] = (rVal - aVal);
	}

	SUM *= 0.5;

	outData[imgID] = SUM;
}


//>*>*>*>**>**>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>*>
///////////////// back propagation ////////////////////
//<*<*<*<*<*<*<*<*<*<*<*<*<*<*<<*<*<*<*<*<*<**<*<*<*<*<*<

__kernel void fc_affine_backP_pass(
	__global float* backInData,
	__global float* backOutData
)
{
	int xID = get_global_id(0); // in-out W
	int imgID = get_global_id(1); // numImage

	int xSz = get_global_size(0);

	int aID = (xSz * imgID) + xID;

	backOutData[aID] = backInData[aID];
}


// back-propagation (batch norm)
__kernel void fc_affine_backP_batchnorm(
	__global float* backInData,
	__global float* prevOutData,
	__global float* prevDEVI,
	__global float* backOutData,
	__global float* calcA,
	int W) // in-out width
{
	int imgID = get_global_id(1);
	int baseID = imgID * W;

	// back sum
	float sumBack = 0.0;
	for (int i = 0; i < W; i++)
	{
		sumBack += backInData[baseID + i] * prevOutData[baseID + i];
	}


	// calc each A
	float DEVI = prevDEVI[imgID];
	float B = 0.0;
	float C = sqrt(DEVI) / (DEVI*DEVI);
	float D = 0.0;
	float m = 1.0 / (float)W;
	float backVal = 0.0;
	float outVal = 0.0;

	for (int i = 0; i < W; i++)
	{
		backVal = backInData[baseID + i];
		outVal = prevOutData[baseID + i];
		B = (backVal / DEVI);
		D = outVal * sumBack;

		calcA[baseID + i] = B - (C * m * D);
	}

	// average of calcA
	float sumA = 0.0;
	for (int i = 0; i < W; i++)
	{
		sumA += calcA[baseID + i];
	}
	sumA /= (float)W;


	// write to back out
	for (int i = 0; i < W; i++)
	{
		backOutData[baseID + i] = calcA[baseID + i] + sumA;
	}
}



__kernel void fc_affine_backP_softmax(
	__global float* backInData,
	__global float* prevEXP,
	__global float* prev_expSum,
	__global float* backOutData,
	int W) // in-out W
{

	int imgID = get_global_id(1);
	int baseID = W * imgID;

	float sumBack_in = 0.0;
	for (int i = 0; i < W; i++)
	{
		sumBack_in += backInData[baseID + i] * prevEXP[baseID + i];
	}

	// -1.0 / S^2
	float S = prev_expSum[imgID];
	sumBack_in /= (S*S);

	// calc each
	for (int i = 0; i < W; i++)
	{
		float prevExpVal = prevEXP[baseID + i];
		float backVal = backInData[baseID + i];

		backOutData[baseID + i] = (backVal / S) - sumBack_in;
	}
}




__kernel void fc_ReLU_back_biasDelta(
	__global float* prevOut, // mem_output_data
	__global float* biasDelta,
	__global float* backInDelta,
	int outW
)
{
	int imgID = get_global_id(1);
	int outBaseID = imgID * outW;

	// check prev out ( used as multiple coef to back-in delta )
	for (int i = 0; i < outW; i++)
	{
		float coef = 0.0;
		if (prevOut[outBaseID + i] != 0.0)
		{
			coef = 1.0;
		}

		// bias delta is back-in value itself
		biasDelta[outBaseID+i] = coef * backInDelta[outBaseID+i];

		// update backInDelta for next weight delta calculation
		backInDelta[outBaseID + i] *= coef;
	}

}

////*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?*?
__kernel void fc_ReLU_sum_biasDelta(
		__global float* bias_each,
		__global float* bias_sum,
		int N
)
{
	int xID = get_global_id(1); // outW
	int outW = get_global_size(1);


	float tempSum = 0.0;

	for (int i = 0; i < N; i++)
	{
		tempSum += bias_each[xID + (i*outW)];
	}

	// write sum
	bias_sum[xID] = tempSum;
}


__kernel void fc_ReLU_back_weightDelta(
	__global float* backIn, // masked
	__global float* prevIn,
	__global float* weightDelta,
	int W // unused in 3D kernel
)
{
	
	int inID = get_global_id(0);
	int outID = get_global_id(1);
	int imgID = get_global_id(2);

	int inW = get_global_size(0);
	int outW = get_global_size(1);

	int inBaseID = imgID * inW;
	int outBaseID = imgID * outW;
	int weiBaseID = imgID * inW * outW;

	// weight delta = back-in * prevInput
	int w_aID = weiBaseID + (inW*outID) + inID;

	weightDelta[w_aID] = backIn[outBaseID + outID] * prevIn[inBaseID + inID];
	

	/* 2D kernel version
	int inID = get_global_id(0);
	int imgID = get_global_id(1);

	int inW = get_global_size(0);

	int outBaseID = imgID * outW;
	int weiBaseID = imgID * (inW * outW);

	float prevInVal = prevIn[(imgID * inW) + inID];

	for (int n = 0; n < outW; n++)
	{
		weightDelta[(weiBaseID + n*inW) + inID] = prevInVal * backIn[outBaseID + n];
	}
	*/
	
}


__kernel void fc_ReLU_sum_weightDelta(
	__global float* weight_each,
	__global float* weight_sum,
	int N
)
{
	int xID = get_global_id(0); // inW
	int yID = get_global_id(1); // outW

	int inW = get_global_size(0);
	int outW = get_global_size(1);
	int imgShift = inW * outW;

	int weiAccID = (inW * yID) + xID;

	// sum weight delta
	float tempSum = 0.0;
	for (int i = 0; i < N; i++)
	{
		tempSum += weight_each[weiAccID + (i*imgShift)];
	}

	weight_sum[weiAccID] = tempSum;
}



__kernel void fc_ReLU_back_out(
	__global float* backIn, // updated by mask
	__global float* weightData, // inW * outW
	__global float* backOut, // inW * numImage
	int outW
)
{
	int xID = get_global_id(0);
	int imgID = get_global_id(1);

	int inW = get_global_size(0);

	int baseID = imgID * inW;
	int backInBaseID = imgID * outW;

	float SUM = 0.0;
	for (int i = 0; i < outW; i++)
	{
		SUM += weightData[inW*i + xID] * backIn[backInBaseID + i];
	}

	backOut[baseID + xID] = SUM;
}



__kernel void fc_CEMS_sum_loss(
	__global float* outData,
	__global float* lossSum,
	int N
)
{
	float sum = 0.0;
	for (int i = 0; i < N; i++)
	{
		sum += outData[i];
	}

	lossSum[0] = sum;
}















