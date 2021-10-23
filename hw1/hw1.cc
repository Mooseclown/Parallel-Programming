#include <cstdio>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

typedef struct Group_info {
	MPI_Group group;
	MPI_Comm comm;
} Group_info;

typedef struct Proc_info {
	int rank;
	int total_proc;		// total process in this program
	float *data;		// subarray
	int len;			// len of data in this rank
	int pre_len;		// len of data in rank - 1
	int next_len;		// len of data in rank + 1
	int first_index;
	int last_index;
	Group_info group_info;
} Proc_info;

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

// if no data to assign to process return 1, else return 0.(success return 0)
int init_group_info(Proc_info *proc_info, int rank, int total_proc, int total_data_len) {
	int ret;
	MPI_Group orig_group, new_group;
	MPI_Comm new_comm;
	MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

	if (total_data_len < total_proc) {
		int *ranks = (int *)malloc(sizeof(int) * total_data_len);
		for (int i = 0; i < total_data_len; i++) {
			ranks[i] = i;
		}

		MPI_Group_incl(orig_group, total_data_len, ranks, &new_group);
		MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

		
		if (rank < total_data_len) {
			proc_info->group_info.group = new_group;
			proc_info->group_info.comm = new_comm;
			ret = 0;
		} else {
			proc_info->group_info.group = orig_group;
			proc_info->group_info.comm = MPI_COMM_WORLD;
			ret = 1;
		}
	} else {
		proc_info->group_info.group = orig_group;
		proc_info->group_info.comm = MPI_COMM_WORLD;
		ret = 0;
	}
	return ret;
}

void init_proc_info(Proc_info *proc_info, int total_proc, int total_data_len) {
	MPI_Comm_rank(proc_info->group_info.comm, &proc_info->rank);
	MPI_Comm_size(proc_info->group_info.comm, &proc_info->total_proc);

	proc_info->len = get_proc_len(proc_info->rank, proc_info->total_proc, total_data_len);
	proc_info->pre_len = get_proc_len(proc_info->rank - 1, proc_info->total_proc, total_data_len);
	proc_info->next_len = get_proc_len(proc_info->rank + 1, proc_info->total_proc, total_data_len);

	proc_info->first_index = proc_info->len * proc_info->rank;
	if (proc_info->rank >= total_data_len % proc_info->total_proc) {
		proc_info->first_index += total_data_len % proc_info->total_proc;
	}
	proc_info->last_index = proc_info->first_index + proc_info->len - 1;
	
	proc_info->data = (float *)malloc(sizeof(float) * proc_info->len);
	return;
}

void free_proc_info(Proc_info *proc_info) {
	free(proc_info->data);
}

void print_array(float *arr, int len, int rank) {
	printf("[%d] ", rank);
	for (int i = 0; i < len; i++) {
		printf("%f ", arr[i]);
	}
	printf("\n");
}

// done return 1. continue return 0.
int is_sort_done(Proc_info *proc_info, unsigned long *is_changed, unsigned long *is_continue, int &double_check) {
	int rc;
	rc = MPI_Allreduce(is_changed, is_continue, 1, MPI_UNSIGNED_LONG, MPI_BOR, proc_info->group_info.comm);
	if (rc != MPI_SUCCESS) printf("error: allreduce error.\n");
	if (*is_continue == 0) {
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
unsigned long cmp_and_swap(float *local_data, float *remote_data, int local_len, int remote_len, int local_is_small_part, int rank) {
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

void odd_even_sort(Proc_info *proc_info) {
	int rc, double_check = 0;
	unsigned long is_continue = 0, is_changed = 0;

	float *pre_buffer, *next_buffer;
	pre_buffer = (float *)malloc(sizeof(float) * proc_info->pre_len);
	next_buffer = (float *)malloc(sizeof(float) * proc_info->next_len);

	std::sort(proc_info->data, proc_info->data + proc_info->len);

	for (int round = 0;; round++) {
		int odd_or_even_phase = round % 2;		// 1 is odd. 0 is even

		MPI_Request send_requset;
		MPI_Status send_status, recv_status;
		if (!(odd_or_even_phase ^ (proc_info->rank % 2))) {
			// send to next process
			if (proc_info->rank != proc_info->total_proc - 1) {
				if (is_changed || (is_continue & 1UL << (proc_info->rank + 1)) || round < 2) {
					rc = MPI_Isend(proc_info->data, proc_info->len, MPI_FLOAT, proc_info->rank + 1, 0, proc_info->group_info.comm, &send_requset);
					if (rc != MPI_SUCCESS) printf("error: send; rank: %d, round: %d.\n", proc_info->rank, round);

					rc = MPI_Recv(next_buffer, proc_info->next_len, MPI_FLOAT, proc_info->rank + 1, MPI_ANY_TAG, proc_info->group_info.comm, &recv_status);
					if (rc != MPI_SUCCESS) printf("error: recv; rank: %d, round: %d.\n", proc_info->rank, round);

					rc = MPI_Wait(&send_requset, &send_status);
					if (rc != MPI_SUCCESS) printf("error: wait send; rank: %d, round: %d.\n", proc_info->rank, round);

					is_changed = cmp_and_swap(proc_info->data, next_buffer, proc_info->len, proc_info->next_len, 1, proc_info->rank);
					is_changed = is_changed << proc_info->rank;
				}
			}
		} else {
			// send to previous process
			if (proc_info->rank != 0) {
				if (is_changed || (is_continue & 1UL << (proc_info->rank - 1)) || round < 2) {
					rc = MPI_Isend(proc_info->data, proc_info->len, MPI_FLOAT, proc_info->rank - 1, 0, proc_info->group_info.comm, &send_requset);
					if (rc != MPI_SUCCESS) printf("error: send; rank: %d, round: %d.\n", proc_info->rank, round);
					
					rc = MPI_Recv(pre_buffer, proc_info->pre_len, MPI_FLOAT, proc_info->rank - 1, MPI_ANY_TAG, proc_info->group_info.comm, &recv_status);
					if (rc != MPI_SUCCESS) printf("error: recv; rank: %d, round: %d.\n", proc_info->rank, round);
					
					rc = MPI_Wait(&send_requset, &send_status);
					if (rc != MPI_SUCCESS) printf("error: wait send; rank: %d, round: %d.\n", proc_info->rank, round);

					is_changed = cmp_and_swap(proc_info->data, pre_buffer, proc_info->len, proc_info->pre_len, 0, proc_info->rank);
					is_changed = is_changed << proc_info->rank;
				}
			}
		}

		is_continue = 0;	// reset check point.
		if (is_sort_done(proc_info, &is_changed, &is_continue, double_check)) {
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
	
	Proc_info proc_info;

	// recreate group
	if (init_group_info(&proc_info, rank, size, atoi(argv[1]))) {
		MPI_Finalize();
		return 0;
	}
	init_proc_info(&proc_info, size, atoi(argv[1]));

	// open and read
	MPI_File f_in;
	rc = MPI_File_open(proc_info.group_info.comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f_in);
	if (rc != MPI_SUCCESS)	printf("error: open read file.\n");

	rc = MPI_File_read_at(f_in, sizeof(float) * proc_info.first_index, proc_info.data, proc_info.len, MPI_FLOAT, MPI_STATUS_IGNORE);
	if (rc != MPI_SUCCESS) printf("error: read file.\n");

	rc = MPI_File_close(&f_in);
	if (rc != MPI_SUCCESS) printf("error: close read file.\n");
	
	odd_even_sort(&proc_info);

	// open and write
	MPI_File f_out;
	rc = MPI_File_open(proc_info.group_info.comm, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f_out);
	if (rc != MPI_SUCCESS)	printf("error: open write file.\n");

	rc = MPI_File_write_at(f_out, sizeof(float) * proc_info.first_index, proc_info.data, proc_info.len, MPI_FLOAT, MPI_STATUS_IGNORE);
	if (rc != MPI_SUCCESS) printf("error: write file.\n");

	MPI_File_close(&f_out);
	if (rc != MPI_SUCCESS) printf("error: close write file.\n");

	free_proc_info(&proc_info);
	MPI_Finalize();
	return 0;
}
