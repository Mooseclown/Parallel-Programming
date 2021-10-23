#include <cstdio>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

typedef struct Proc_info {
	int rank;
	int total_proc;		// total process in this program
	float *data;		// subarray
	int len;			// len of data in this rank
	int pre_len;		// len of data in rank - 1
	int next_len;		// len of data in rank + 1
	int first_index;
	int last_index;
	int valid;			// process dose not have any data to sort
} proc_info;

int get_proc_len(int rank, int total_proc, int total_data_len) {
	if (rank < 0 || rank > total_proc - 1) {
		return 0;
	}

	int len;
	len = total_data_len / total_proc;
	if (rank < total_data_len % total_proc) {
		len += 1;
	}
	return len;
}

void init_proc_info(proc_info *info, int rank, int total_proc, int total_data_len) {
	info->rank = rank;
	info->total_proc = total_proc;
	info->len = get_proc_len(rank, total_proc, total_data_len);
	info->pre_len = get_proc_len(rank - 1, total_proc, total_data_len);
	info->next_len = get_proc_len(rank + 1, total_proc, total_data_len);

	info->first_index = info->len * info->rank;
	if (rank >= total_data_len % total_proc) {
		info->first_index += total_data_len % total_proc;
	}
	info->last_index = info->first_index + info->len - 1;
	
	info->data = (float *)malloc(sizeof(float) * info->len);
	
	if (total_data_len < total_proc) {
		info->total_proc = total_data_len;
	}

	if (info->len == 0) {
		info->valid = 0;
	} else {
		info->valid = 1;
	}
}

void free_proc_info(proc_info *info) {
	free(info->data);
}

void print_array(float *arr, int len, int rank) {
	printf("[%d] ", rank);
	for (int i = 0; i < len; i++) {
		printf("%f ", arr[i]);
	}
	printf("\n");
}

int is_sort_done(int is_changed, int &double_check) {
	int is_continue, rc;
	rc = MPI_Allreduce(&is_changed, &is_continue, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
	if (rc != MPI_SUCCESS) printf("error: allreduce error.\n");
	if (!is_continue) {
		if (double_check) {
			return 1;
		}
		double_check = 1;
	}
	else {
		double_check = 0;
	}
	return 0;
}

// if no exchange return 0, else return 1
int cmp_and_swap(float *local_data, float *remote_data, int local_len, int remote_len, int local_is_small_part, int rank) {
	int local_index, remote_index, tmp_index;
	float *tmp_data = (float *)malloc(sizeof(float) * local_len);

	if (local_is_small_part) {
		local_index = 0;
		remote_index = 0;
		tmp_index = 0;

		while (tmp_index < local_len) {
			if (remote_index > remote_len - 1 || local_data[local_index] <= remote_data[remote_index]) {
				tmp_data[tmp_index++] = local_data[local_index++];
			} else {
				tmp_data[tmp_index++] = remote_data[remote_index++];
			}
		}
	} else {
		local_index = local_len - 1;
		remote_index = remote_len - 1;
		tmp_index = local_len - 1;

		while (tmp_index >= 0) {
			if (remote_index < 0 || local_data[local_index] >= remote_data[remote_index]) {
				tmp_data[tmp_index--] = local_data[local_index--];
			} else {
				tmp_data[tmp_index--] = remote_data[remote_index--];
			}
		}
	}
	
	std::copy(tmp_data, tmp_data + local_len, local_data);
	if (local_index == tmp_index) {
		return 0;
	} else {
		return 1;
	}
}

void odd_even_sort(proc_info *info) {
	int rc, double_check = 0;

	if (!info->valid) {
		while (!is_sort_done(0, double_check));
		return;
	}

	float *pre_buffer, *next_buffer;
	pre_buffer = (float *)malloc(sizeof(float) * info->pre_len);
	next_buffer = (float *)malloc(sizeof(float) * info->next_len);

	std::sort(info->data, info->data + info->len);

	for (int round = 0;; round++) {
		int odd_or_even_phase = round % 2;		// 1 is odd. 0 is even
		int is_changed = 0;

		MPI_Request send_requset;
		MPI_Status send_status, recv_status;
		if (!(odd_or_even_phase ^ (info->rank % 2))) {
			// send to next process
			if (info->rank != info->total_proc - 1) {
				rc = MPI_Isend(info->data, info->len, MPI_FLOAT, info->rank + 1, 0, MPI_COMM_WORLD, &send_requset);
				if (rc != MPI_SUCCESS) printf("error: send; rank: %d, round: %d.\n", info->rank, round);

				rc = MPI_Recv(next_buffer, info->next_len, MPI_FLOAT, info->rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_status);
				if (rc != MPI_SUCCESS) printf("error: recv; rank: %d, round: %d.\n", info->rank, round);

				rc = MPI_Wait(&send_requset, &send_status);
				if (rc != MPI_SUCCESS) printf("error: wait send; rank: %d, round: %d.\n", info->rank, round);

				is_changed = cmp_and_swap(info->data, next_buffer, info->len, info->next_len, 1, info->rank);
			}
		} else {
			// send to previous process
			if (info->rank != 0) {
				rc = MPI_Isend(info->data, info->len, MPI_FLOAT, info->rank - 1, 0, MPI_COMM_WORLD, &send_requset);
				if (rc != MPI_SUCCESS) printf("error: send; rank: %d, round: %d.\n", info->rank, round);
				
				rc = MPI_Recv(pre_buffer, info->pre_len, MPI_FLOAT, info->rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_status);
				if (rc != MPI_SUCCESS) printf("error: recv; rank: %d, round: %d.\n", info->rank, round);
				
				rc = MPI_Wait(&send_requset, &send_status);
				if (rc != MPI_SUCCESS) printf("error: wait send; rank: %d, round: %d.\n", info->rank, round);

				is_changed = cmp_and_swap(info->data, pre_buffer, info->len, info->pre_len, 0, info->rank);
			}
		}
		
		// check if it's done
		if (is_sort_done(is_changed, double_check)) {
			break;
		}
	}
	free(pre_buffer);
	free(next_buffer);
	return;
}

int main(int argc, char** argv) {
	MPI_Init(&argc,&argv);
	int rank, size, rc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	proc_info info;

	init_proc_info(&info, rank, size, atoi(argv[1]));

	// open and read
	MPI_File f_in;
	rc = MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f_in);
	if (rc != MPI_SUCCESS)	printf("error: open read file.\n");

	if (info.valid) {
		rc = MPI_File_read_at(f_in, sizeof(float) * info.first_index, info.data, info.len, MPI_FLOAT, MPI_STATUS_IGNORE);
		if (rc != MPI_SUCCESS) printf("error: read file.\n");
	}

	rc = MPI_File_close(&f_in);
	if (rc != MPI_SUCCESS) printf("error: close read file.\n");
	
	odd_even_sort(&info);

	// open and write
	MPI_File f_out;
	rc = MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f_out);
	if (rc != MPI_SUCCESS)	printf("error: open write file.\n");

	if (info.valid) {
		rc = MPI_File_write_at(f_out, sizeof(float) * info.first_index, info.data, info.len, MPI_FLOAT, MPI_STATUS_IGNORE);
		if (rc != MPI_SUCCESS) printf("error: write file.\n");
	}
	MPI_File_close(&f_out);
	if (rc != MPI_SUCCESS) printf("error: close write file.\n");

	free_proc_info(&info);
	MPI_Finalize();
	return 0;
}