#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NUM_THREADS		4
#define CHUNK	1

#define ULL_MAX		~0ULL

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long square_r = r*r;
	unsigned long long limit = k <= r ? ULL_MAX - r : k;

	omp_set_num_threads(NUM_THREADS);

	#pragma omp parallel for schedule(static, CHUNK) reduction(+: pixels)
	for (unsigned long long x = 0; x < r; x++) {
		unsigned long long y = ceil(sqrtl(square_r - x*x));
		pixels += y;
		if (pixels >= limit)
			pixels %= k;
	}

	printf("%llu\n", (4 * pixels) % k);
}
