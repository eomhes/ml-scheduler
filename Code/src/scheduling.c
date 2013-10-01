#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

#include "scheduling.h"


typedef struct
{
	double comptocomm;
	int num_setargs;
	int label;
} training_set;

training_set repository[NUM_INSTANCES];

//double offload_ctoc[NUM_INSTANCES] = {13.55, 5.22, 1.87, 13.97, 13.04, 4.41, 1.97, 1.81, 12.32, 9.23, 4.84, 12.95, 13.18, 5.00, 12.95, 1.81, 32.91, 12.34, 42.13, 10.10, 17.34, 5.35, 21.15, 5.33, 31.03, 11.35, 3.57, 10.87, 13.62, 6.93, 23.39, 5.20, 11.40, 4.75, 15.25, 3.52, 74.79, 28.59, 9.97, 2.12, 36.77, 16.61, 5.96, 1.32, 17.81, 7.07, 18.62, 17.76, 24.48, 22.07, 20.35, 8.87, 24.34, 18.23, 24.86, 25.73, 23.42, 25.06, 24.53, 25.06, 25.42, 25.73, 24.09, 24.19};
//double local_ctoc[NUM_INSTANCES] = {0.42, 1.76, 0.42, 0.38, 0.42, 1.76, 1.74, 2.62, 12.69, 8.83, 10.55, 2.62, 1.63, 0.35, 0.33, 1.16, 1.76, 1.74, 1.61, 1.63, 2.36, 1.93, 2.20 ,2.44, 1.93, 2.46, 1.94, 0.53, 0.50, 0.55, 0.54, 0.55, 3.34, 2.88, 0.73, 0.74, 8.06, 9.60, 3.32, 0.75, 0.64, 9.80, 3.41, 0.75, 0.71, 24.86, 8.79, 3.13, 0.38, 0.41, 0.40, 0.40, 0.41, 2.36, 2.20, 0.53, 0.50, 0.53, 9.02, 0.75, 3.33, 9.80, 3.41, 0.71};
//int offload_setargs[NUM_INSTANCES] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 400, 400, 400, 400, 200, 200, 200, 200, 40, 40, 40, 40, 20, 20, 20, 20, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
//int local_setargs[NUM_INSTANCES] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 1000, 1000, 1000, 1000, 1000, 1000};

double training_ctoc[NUM_INSTANCES] = {13.55, 13.84, 13.97, 5.22, 5.16, 4.90, 1.87, 1.85, 1.88, 0.41, 0.41, 0.42, 12.48, 14.08, 11.60, 5.23, 5.28, 5.22, 1.76, 1.84, 1.84, 0.40, 0.38, 0.38, 12.35, 12.22, 12.57, 4.78, 4.87, 5.03, 1.74, 1.72, 1.71, 0.39, 0.38, 0.38, 9.77, 7.23, 11.17, 4.25, 3.66, 3.48, 1.54, 1.55, 1.67, 0.35, 0.38, 0.32, 36341.62, 35489.63, 32543.63, 13487.10, 4282.15, 4560.16, 4624.53, 997.90, 1033.66, 1048.50, 16163.97, 14458.60, 17981.93, 6451.41, 6506.58, 6190.74, 2156.56, 2286.40, 2373.73, 514.39, 525.09, 529.70, 2834.71, 3051.71, 3635.97, 1145.18, 1066.26, 1259.24, 458.02, 408.77, 368.07, 98.38, 104.64, 106.13, 1545.49, 1523.17, 1623.67, 511.59, 668.88, 213.58, 237.14, 229.95, 49.12, 53.29, 47.63, 118.48, 116.30, 117.05, 35.00, 40.10, 44.91, 11.81, 12.98, 14.75, 3.30, 3.51, 3.39, 75.28, 73.38, 64.68, 26.78, 28.75, 28.15, 9.22, 9.77, 9.86, 2.24, 2.23, 2.23, 35.98, 40.47, 42.58, 17.04, 16.98, 16.78, 4.64, 4.38, 5.83, 1.32, 1.25, 1.24, 17.36, 12.07, 18.02, 7.00, 6.13, 2.46, 2.42, 2.44, 0.52, 0.55, 0.51, 25.09, 23.18, 23.72, 9.85, 9.62, 9.47, 3.37, 3.36, 3.36, 0.72, 0.75, 0.75, 24.38, 24.86, 25.32, 9.60, 8.88, 8.72, 3.06, 3.35, 3.36, 0.74, 0.74, 0.61, 24.53, 19.59, 9.20, 7.42, 9.19, 2.73, 3.33, 3.30, 0.73, 0.69, 0.69, 13.10, 23.29, 22.37, 7.42, 8.23, 8.60, 3.29, 2.59, 3.15, 0.73, 0.72, 0.70};

int training_setargs[NUM_INSTANCES] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};

int training_class[NUM_INSTANCES] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

bool offline_scheduler_RT_old(double ctoc, double bandwidth)
{
	bool sched_offload = false;
	
	if (ctoc < 4.45) {
		if (bandwidth < 1.09) {
			if (bandwidth < 0.91) {
				sched_offload = false;
			}
			else {
				if (ctoc < 2.07) {
					if (ctoc < 1.71) {
						sched_offload = false;
					}
					else {
						sched_offload = true;
					}
				}
				else {
					sched_offload = false;
				}
			}
		}
		else {
			if (ctoc < 1.64) {
				sched_offload = false;
			}
			else {
				if (ctoc < 2.74) {
					sched_offload = true;
				}
				else {
					if (ctoc < 3.15) {
						sched_offload = false;
					}
					else {
						if (bandwidth < 4.32) {
							if (ctoc < 3.2) {
								sched_offload = false;
							}
							else {
								sched_offload = true;
							}
						}
						else {
							sched_offload = true;
						}
					}
				}
			}
		}
	}
	else {
		sched_offload = true;
	}
	
	return sched_offload;
}

bool offline_scheduler_RT_new(double ctoc, int num_setargs) 
{
	bool sched_offload;
	
	if (ctoc < 9.88) {
		if (ctoc < 1.21) {
			sched_offload = false;
		}
		else {
			if (num_setargs < 504) {
				if (ctoc < 4.7) {
					if (ctoc < 1.34) {
						sched_offload = true;
					}
					else {
						if (ctoc < 1.78) {
							if (ctoc < 1.74) {
								if(ctoc < 1.64) {
									sched_offload = false;
								}
								else {
									if (ctoc < 1.65) {
										sched_offload = true;
									}
									else {
										if (ctoc < 1.67) {
											sched_offload = true;
										}
										else {
											if (ctoc < 1.69) {
												sched_offload = false;
											}
											else {
												if (ctoc < 1.71) {
													sched_offload = false;
												}
												else {
													sched_offload = true;
												}
											}
										}
									}
								}
							}
							else {
								sched_offload = false;
							}
						}
						else {
							if (ctoc < 4.47) {
								if (ctoc < 1.9) {
									sched_offload = true;
								}
								else {
									if (ctoc < 1.96) {
										sched_offload = false;
									}
									else {
										if (ctoc < 2.16) {
											sched_offload = true;
										}
										else {
											if (ctoc < 3.46) {
												if (ctoc < 2.3) {
													if (ctoc < 2.21) {
														sched_offload = false;
													}
													else {
														sched_offload = true;
													}
												}
												else {
													sched_offload = false;
												}
											}
											else {
												if (num_setargs < 7.5) {
													if (ctoc < 3.81) {
														sched_offload = false;
													}
													else {
														sched_offload = true;
													}
												}
												else {
													sched_offload = true;
												}
											}
										}
									}
								}
							}
							else {
								if (ctoc < 7.11) {
									if (ctoc < 6.36) {
										sched_offload = true;
									}
									else {
										if (ctoc < 6.89) {
											sched_offload = false;
										}
										else {
											sched_offload = true;
										}
									}
								}
								else {
									if (ctoc < 9.03) {
										sched_offload = false;
									}
									else {
										if (ctoc < 9.3) {
											sched_offload = true;
										}
										else {
											if (ctoc < 9.38) {
												sched_offload = false;
											}
											else {
												sched_offload = true;
											}
										}
									}
								}
							}
						}
					}
				}
			}
			else {
				if (ctoc < 8.43) {
					sched_offload = false;
				}
				else {
					if (ctoc < 8.76) {
						sched_offload = true;
					}
					else {
						if (ctoc < 9.64) {
							if (ctoc < 9.54) {
								if (ctoc < 8.9) {
									if (ctoc < 8.83) {
										sched_offload = false;
									}
									else {
										sched_offload = true;
									}
								}
								else {
									sched_offload = false;
								}

							}
							else {
								if (ctoc < 9.59) {
									if (ctoc < 9.58) {
										sched_offload = true;
									}
									else {
										sched_offload = true;
									}
								}
								else {
									if (ctoc < 9.63) {
										sched_offload = false;
									}
									else {
										sched_offload = true;
									}
								}
							}
						}
						else {
							sched_offload = false;
						}
					}
				}
			}
		}
	}
	else {
		if (num_setargs < 700) {
			if (ctoc < 12.82) {
				if (ctoc < 12.56) {
					if (ctoc < 10.47) {
						sched_offload = true;
					}
					else {
						if (ctoc < 11.36) {
							sched_offload = false;
						}
						else {
							sched_offload = true;
						}
					}
				}
				else {
					sched_offload = false;
				}
			}
			else {
				sched_offload = true;
			}
		}
		else {
			if (ctoc < 25.04) {
				if (ctoc < 24.94) {
					if (ctoc < 19) {
						sched_offload = true;
					}
					else {
						if (ctoc < 20.18) {
							sched_offload = false;
						}
						else {
							if (ctoc < 20.58) {
								if (ctoc < 21.44) {
									sched_offload = false;
								}
								else {
									if (ctoc < 23.61) {
										sched_offload = true;
									}
									else {
										if (ctoc < 23.99) {
											sched_offload = false;
										}
										else {
											if (ctoc < 24.7) {
												sched_offload = true;
											}
											else {
												sched_offload = true;
											}
										}
									}
								}
							}
						}
					}
				}
				else {
					sched_offload = false;
				}
			}
			else {
				sched_offload = true;
			}
		}
	}
	return sched_offload;
}

unsigned int
randr(unsigned int min, unsigned int max)
{
	double scaled = (double)rand()/RAND_MAX;
	
	return (max - min + 1) * scaled + min;
}

bool offline_scheduler_IBL(double ctoc, int num_setargs)
{
	bool sched_offload;
	int i, j;
	double min_distance_to_instance = 10000000;
	double distance_to_instance;
	int min_distance_index;

	for (i = 0; i < NUM_INSTANCES; i++) {
		//j = randr(0, NUM_INSTANCES);
		distance_to_instance = sqrt((double) pow(training_ctoc[i] - ctoc, 2) 
										+(double) pow(training_setargs[i] - num_setargs, 2));
		
		if (distance_to_instance < min_distance_to_instance) {
			min_distance_to_instance = distance_to_instance;
			min_distance_index = i;
		}
	}
	
	//printf("index %d\n", min_distance_index);
	if (training_class[min_distance_index] == 0) {
		sched_offload = false;
	}
	else {
		sched_offload = true;
	}
	
	return sched_offload;
}

bool online_scheduler_IBL(double ctoc, int num_setargs)
{
	bool sched_offload;
	int i, j;
	double min_distance_to_instance = 10000000;
	double distance_to_instance;
	int min_distance_index;

	for (i = 0; i < MAX_TRAINING; i++) {
		distance_to_instance = sqrt((double) pow(repository[i].comptocomm - ctoc, 2)
		                                +(double) pow(repository[i].num_setargs - num_setargs, 2));
		if (distance_to_instance < min_distance_to_instance) {
			min_distance_to_instance = distance_to_instance;
			min_distance_index = i;
		}
	}
	printf("index %d\n", min_distance_index);
	if (repository[min_distance_index].label == 0) {
		sched_offload = false;
		printf("0\n");
	}
	else {
		sched_offload = true;
		printf("1\n");
	}

	return sched_offload;
}

bool heuristic_scheduler_threshold(double ctoc, double threshold)
{
	 bool sched_offload;
	 
	 if (ctoc < threshold) {
		 sched_offload = false;
	 }
	 else {
		 sched_offload = true;
	 }
	 
	 return sched_offload;
}

void put_instance(double ctoc, int net_setargs, bool label, int index)
{
	//printf("index number: %d\n", index);
	repository[index].comptocomm = ctoc;
	repository[index].num_setargs = net_setargs;
	repository[index].label = label;
}
