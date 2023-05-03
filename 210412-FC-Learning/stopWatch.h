#pragma once
class stopWatch
{
public:

	LARGE_INTEGER freq; // counter's frequency
	LARGE_INTEGER start_time;
	LARGE_INTEGER end_time;
	double SECOND;

	stopWatch();
	~stopWatch();


	void start_stop_watch();
	void finish_stop_watch();
};

