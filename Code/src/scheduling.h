#ifndef _SCHEDULING_H_
#define _SCHEDULING_H_

#define NUM_INSTANCES 192
#define IB_CMP_NUM 50
#define MAX_TRAINING 16

#endif

bool offline_scheduler_RT_old(double ctoc,
						      double bandwidth);

bool offline_scheduler_RT_new(double ctoc,
							  int num_setargs);

unsigned int randr(unsigned int min, 
				   unsigned int max);

bool offline_scheduler_IBL(double ctoc, 
						   int net_setargs);

bool online_scheduler_IBL(double ctoc,
					      int net_setargs);

bool heuristic_scheduler_threshold(double ctoc,
					               double threshold);

void put_instance(double ctoc, 
		          int net_setargs, 
				  bool label,
				  int index);
