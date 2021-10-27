#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <sched.h>
#include <pthread.h>

#define ULL_MAX		~0ULL

typedef struct thread_info {
	int tid;
	int ncpus;
	unsigned long long r;
	unsigned long long k;
} thread_info;

void *compute(void *info) {
	thread_info *t_info = (thread_info *)info;
	
	int cpu_id = t_info->tid;
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(cpu_id, &cpuset);
	pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

	unsigned long long *sum = (unsigned long long *)malloc(sizeof(unsigned long long));
	unsigned long long square_r = t_info->r*t_info->r;
	unsigned long long limit = ULL_MAX - t_info->r;
	
	for (unsigned long long x = t_info->tid; x < t_info->r; x += t_info->ncpus) {
		unsigned long long y = ceil(sqrtl(square_r - x*x));
		*sum += y;
		if (*sum >=  limit)
			*sum %= t_info->k;
	}
	pthread_exit((void *)sum);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;

	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);

	pthread_t threads[ncpus];
	thread_info t_info[ncpus];

	for (unsigned long long i = 0; i < ncpus; i++) {
		int rc;
		t_info[i].tid = i;
		t_info[i].ncpus = ncpus;
		t_info[i].r = r;
		t_info[i].k = k;
		rc = pthread_create(&threads[i], NULL, compute, (void *)&t_info[i]);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}
	
	void *ret;
	unsigned long long result;
	for (unsigned long long i = 0; i < ncpus; i++) {
		pthread_join(threads[i], &ret);
		result = *(unsigned long long *)ret;
		pixels += result;
		pixels %= k;
		free((unsigned long long *)ret);
	}
	printf("%llu\n", (4 * pixels) % k);
}
