#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define ULL_MAX 	~0ULL
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	int rank, size;
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // the total number of process
	MPI_Comm_size(MPI_COMM_WORLD, &size); // the rank (id) of the calling process

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long total_pixels = 0;
	unsigned long long square_r = r*r;
	unsigned long long limit = k <= r ? ULL_MAX - r : k;

	for (unsigned long long x = rank; x < r; x += size) {
		unsigned long long y = ceil(sqrtl(square_r - x*x));
		pixels += y;
		if (pixels >= limit)
			pixels %= k;
	}
	MPI_Reduce(&pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG , MPI_SUM, size - 1, MPI_COMM_WORLD);
	if (rank == size - 1) {
		printf("%llu\n", (4 * total_pixels) % k);
	}
	MPI_Finalize();
	return 0;
}
