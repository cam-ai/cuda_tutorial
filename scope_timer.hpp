#ifndef SCOPETIMER__H
#define SCOPETIMER__H

#include <iostream>
#include <chrono>

enum class TimerPlatForm
{
	CPU = 1,
	GPU = 2,
};

template <TimerPlatForm platform>
struct ScopeTimer
{
	std::chrono::high_resolution_clock::time_point start;
	cudaEvent_t event_start, event_end;
	float elapsed = 0.0f;
	const char* title;
	ScopeTimer()
	{
		title = nullptr;
		if constexpr (platform == TimerPlatForm::CPU) {
			start = std::chrono::high_resolution_clock::now();
		} else if constexpr (platform == TimerPlatForm::GPU) {
			cudaEventCreate(&event_start);
    		cudaEventCreate(&event_end);
			cudaEventRecord(event_start);
		}
	}

	ScopeTimer(const char* title)
	{
		this->title = title;
		if constexpr (platform == TimerPlatForm::CPU) {
			start = std::chrono::high_resolution_clock::now();
		} else if constexpr (platform == TimerPlatForm::GPU) {
			cudaEventCreate(&event_start);
    		cudaEventCreate(&event_end);
			cudaEventRecord(event_start);
		}
	}

	~ScopeTimer()
	{
		using std::chrono::duration_cast;
		using std::chrono::nanoseconds;
        using std::chrono::milliseconds;

		if constexpr (platform == TimerPlatForm::CPU) {
			auto end = std::chrono::high_resolution_clock::now();
			auto duration_ns = duration_cast<nanoseconds>(end - start);
        	auto duration_ms = duration_cast<milliseconds>(end - start);
			std::cout << (title ? title : "") << ": " << duration_ns.count() << " ns\t" << duration_ms.count() << " ms\n";
		} else if constexpr (platform == TimerPlatForm::GPU) {
			cudaEventRecord(event_end);
			cudaEventSynchronize(event_end);
        	cudaEventElapsedTime(&elapsed, event_start, event_end);
			std::cout << (title ? title : "") << ": " <<  elapsed << " ms\n";
			cudaEventDestroy(event_start);
			cudaEventDestroy(event_end);
		}
	}
};

#endif