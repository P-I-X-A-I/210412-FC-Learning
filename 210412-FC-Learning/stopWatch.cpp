#include "framework.h"
#include "stopWatch.h"


stopWatch::stopWatch()
{

}


stopWatch::~stopWatch()
{
}




void stopWatch::start_stop_watch()
{
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start_time);

}



void stopWatch::finish_stop_watch()
{
	QueryPerformanceCounter(&end_time);
	
	LONGLONG span = end_time.QuadPart - start_time.QuadPart;

	SECOND = (double)span / (double)freq.QuadPart;
}
