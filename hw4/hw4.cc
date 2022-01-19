#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <pthread.h>
#include <assert.h>
#include <unistd.h>
#include <ctime>

int rank, num_nodes;
unsigned long ncpus;
int num_reducer, delay, chunk_size, num_chunk;
std::string job_name, input_filename, locality_config_filename, output_dir;

std::string metadata_dir = "/home/pp21/pp21s42/pp/hw4/metadata/";

typedef std::pair<std::string, int> Item;
typedef std::pair<int, std::string> Record;

enum MPI_TAG {
    TASK_REQUEST_CHUNK_ID = 1,
    TASK_RETURN_MATADATA_NAME = 2,
    TASK_REQUEST_REDUCE_ID = 3,
    JOB_RETURN_REDUCE_ID = 4,
    TASK_RETURN_COMPELTE = 5,
    MAX_MPI_TAG = 6
};

int get_chunkID(int cpu_id, unsigned long &start_time) {
    int mapperID = (rank - 1) * (ncpus - 1) + cpu_id + 1, chunkID;
    MPI_Status recv_status;

    /* send node ID */
    MPI_Send(&mapperID, 1, MPI_INT, 0, TASK_REQUEST_CHUNK_ID, MPI_COMM_WORLD);

    /* recv chunk ID */
    MPI_Recv(&chunkID, 1, MPI_INT, 0, mapperID + MAX_MPI_TAG, MPI_COMM_WORLD, &recv_status);
    MPI_Recv(&start_time, 1, MPI_UNSIGNED_LONG, 0, mapperID + MAX_MPI_TAG, MPI_COMM_WORLD, &recv_status);

    return chunkID;
}

void spilt_input(int chunkID, std::map<int, std::string> &record) {
    std::ifstream input_file(input_filename);
    std::string line;
    int line_num = 0;
    while (line_num < chunkID * chunk_size && getline(input_file, line)) {
        ++line_num;
    }

    for (int i = 0; i < chunk_size; ++i) {
        getline(input_file, line);
        record[line_num + i] = line;
    }
    input_file.close();
}

void map(std::map<int, std::string> &record, std::map<std::string, int> &word_count) {
    for (auto &item : record) {
        std::string line = item.second;
        size_t pos = 0;
        std::string word;
        std::vector<std::string> words;
        while ((pos = line.find(" ")) != std::string::npos)
        {
            word = line.substr(0, pos);
            words.push_back(word);

            line.erase(0, pos + 1);
        }
        if (!line.empty())
            words.push_back(line);
        for (auto word : words)
        {
            if (word_count.count(word) == 0)
            {
                word_count[word] = 1;
            }
            else
            {
                word_count[word]++;
            }
        }
    }
}

int partition(std::string word) {
    std::hash<std::string> hash_obj;

    return hash_obj(word) % num_reducer + 1;
}

void send_reduce_filename(std::map<int, std::vector<Item>> reduce_files,
                          int chunkID, int num_K_V_pair, unsigned long &start_time) {
    int info[3];
    info[0] = reduce_files.size();
    info[1] = chunkID;
    info[2] = num_K_V_pair;
    MPI_Send(info, 3, MPI_INT, 0, TASK_RETURN_MATADATA_NAME, MPI_COMM_WORLD);
    
    int reducerIDs[info[0]];
    int i = 0;
    for (auto &reduce_file: reduce_files) {
        reducerIDs[i++] = reduce_file.first;
    }
    MPI_Send(reducerIDs, info[0], MPI_INT, 0, chunkID + MAX_MPI_TAG * 2, MPI_COMM_WORLD);
    MPI_Send(&start_time, 1, MPI_UNSIGNED_LONG, 0, TASK_RETURN_MATADATA_NAME, MPI_COMM_WORLD);
}

void *mapper_task(void *opaque) {
    /* set cpu core */
    int cpu_id = *(int *)opaque;
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(cpu_id, &cpu_set);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
    
    int chunkID, reduceID;

    while (1) {
        std::map<int, std::string> record;
        std::map<std::string, int> word_count;
        unsigned long start_time;
        chunkID = get_chunkID(cpu_id, start_time);

        if (chunkID == 0) {
            break;
        } else if (chunkID < 0) {
            chunkID = chunkID * -1;
            sleep(delay);
        }

        spilt_input(chunkID - 1, record);

        map(record, word_count);

        /* partition word to reduce files */
        std::map<int, std::vector<Item>> reduce_files;
        for (auto &item : word_count) {
            reduceID = partition(item.first);
            if (reduce_files.count(reduceID) == 0) {
                std::vector<Item> wordcount_list;
                reduce_files[reduceID] = wordcount_list;
            }
            reduce_files[reduceID].push_back(item);
        }

        /* write reduce files */
        int num_K_V_pair = 0;
        for (auto &reduce_file : reduce_files) {
            std::ofstream out(metadata_dir + std::to_string(reduce_file.first) +
                              "-" + std::to_string(chunkID) + "-metadata.out");

            num_K_V_pair += reduce_file.second.size();
            for (auto &item : reduce_file.second) {
                out << item.first << " " << item.second << "\n";
            }
            out.close();
        }

        send_reduce_filename(reduce_files, chunkID, num_K_V_pair, start_time);
    }
    pthread_exit(NULL);
}

int get_reduceID(unsigned long &start_time) {
    int reducerID = rank;
    MPI_Send(&reducerID, 1, MPI_INT, 0, TASK_REQUEST_REDUCE_ID, MPI_COMM_WORLD);

    int reduceID;
    MPI_Status recv_status;
    MPI_Recv(&reduceID, 1, MPI_INT, 0, JOB_RETURN_REDUCE_ID, MPI_COMM_WORLD,
             &recv_status);

    MPI_Recv(&start_time, 1, MPI_UNSIGNED_LONG, 0, JOB_RETURN_REDUCE_ID,
             MPI_COMM_WORLD, &recv_status);

    return reduceID;
}

void read_reduce_file(int reduceID, std::vector<Item> &wordcount_list) {
    std::ifstream reduece_file(metadata_dir + job_name + '-' +
                               std::to_string(reduceID) + "-metadata.out");

    std::string line;
    while (getline(reduece_file, line)) {
        size_t pos = 0;
        std::string word, count;
        pos = line.find(" ");
        word = line.substr(0, pos);
        count = line.substr(pos + 1);
        Item word_count(word, std::stoi(count));
        wordcount_list.push_back(word_count);
    }
}

void sort(std::vector<Item> &wordcount_list) {
    std::sort(wordcount_list.begin(), wordcount_list.end(), [](const Item &item1, const Item &item2) -> bool
              { return item1.first < item2.first; });
}

void group(std::vector<Item> &wordcount_list, std::map<std::string, std::vector<int>> &word_counts) {
    for (auto word_count : wordcount_list) {
        if (word_counts.count(word_count.first) == 0) {
            std::vector<int> counts;
            word_counts[word_count.first] = counts;
        }
        word_counts[word_count.first].push_back(word_count.second);
    }
}

void reduce(std::map<std::string, std::vector<int>> &word_counts, std::map<std::string, int> &word_count) {
    int total_count;
    for (auto item : word_counts) {
        total_count = 0;
        for (auto count : item.second) {
            total_count += count;
        }
        word_count[item.first] = total_count;
    }
}

void output(int reduceID, std::map<std::string, int> &word_count) {
    std::ofstream out(output_dir + "/" + job_name + "-" + std::to_string(reduceID) + ".out");
    for (auto item : word_count) {
        out << item.first << " " << std::to_string(item.second) << "\n";
    }
}

void *reducer_task(void *opaque) {
    /* set cpu core */
    int cpu_id = *(int *)opaque;
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(cpu_id, &cpu_set);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
    
    int reduceID;
    while (1) {
        unsigned long start_time;
        reduceID = get_reduceID(start_time);
        if (reduceID == 0) {
            break;
        }

        std::vector<Item> wordcount_list;
        read_reduce_file(reduceID, wordcount_list);

        sort(wordcount_list);

        std::map<std::string, std::vector<int>> word_counts;
        group(wordcount_list, word_counts);

        std::map<std::string, int> word_count;
        reduce(word_counts, word_count);

        output(reduceID, word_count);

        /* return task finish */
        MPI_Send(&reduceID, 1, MPI_INT, 0, TASK_RETURN_COMPELTE,
                 MPI_COMM_WORLD);

        MPI_Send(&start_time, 1, MPI_UNSIGNED_LONG, 0, TASK_RETURN_COMPELTE,
                 MPI_COMM_WORLD);
    }
    pthread_exit(NULL);
}

void tasktracker() {
    /* create task threads */
    pthread_t mapper_thread[ncpus - 1], reducer_thread;

    int cpu_id[ncpus];
    for (int i = 0; i < ncpus - 1; ++i) {
        cpu_id[i] = i;
        pthread_create(&mapper_thread[i], NULL, mapper_task, (void *)&cpu_id[i]);
    }
    
    for (int i = 0; i < ncpus - 1; ++i) {
		pthread_join(mapper_thread[i], NULL);
	}

    cpu_id[ncpus - 1] = ncpus - 1;
    pthread_create(&reducer_thread, NULL, reducer_task, (void *)&cpu_id[ncpus - 1]);
    pthread_join(reducer_thread, NULL);
}

void read_locality_config_file(std::map<int, int> &locality_config) {
    std::ifstream locality_config_file(locality_config_filename);
    std::string line;
    while (getline(locality_config_file, line)) {
        size_t pos = 0;
        std::string chunkID, nodeID;
        pos = line.find(" ");
        chunkID = line.substr(0, pos);
        nodeID = line.substr(pos + 1);
        locality_config[std::stoi(chunkID)] = std::stoi(nodeID);
    }
}

int get_chunk_by_locality_config(std::map<int, int> &locality_config, int nodeID) {
    int chunkID;
    for (auto &item : locality_config) {
        if (item.second == nodeID) {
            chunkID = item.first;
            locality_config.erase(chunkID);
            return chunkID;
        }
    }

    for (auto &item : locality_config) {
        chunkID = item.first;
        locality_config.erase(chunkID);
        return chunkID * -1;
    }
    return 0;
}

void *recv_reduce_filename(void *opaque) {
    /* set cpu core */
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(1, &cpu_set);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);

    std::ofstream &outfile = *(std::ofstream *)opaque;

    int recv_chunk_num = 0;
    int chunkID, nodeID;
    int num_K_V_pair = 0;
    std::map<int, std::vector<int>> reducer_files;

    while (recv_chunk_num < num_chunk) {

        /* recv reduce filename */
        int info[3]; // 0 is number of files, 1 is chunk ID, 2 is num of K-V pair.
        MPI_Status recv_status;
        MPI_Recv(&info, 3, MPI_INT, MPI_ANY_SOURCE, TASK_RETURN_MATADATA_NAME, MPI_COMM_WORLD, &recv_status);
        
        nodeID = recv_status.MPI_SOURCE;
        chunkID = info[1];
        num_K_V_pair += info[2];

        int reducerIDs[info[0]];
        MPI_Recv(reducerIDs, info[0], MPI_INT, nodeID, chunkID + MAX_MPI_TAG * 2, MPI_COMM_WORLD,
                 &recv_status);

        unsigned long start_time;
        MPI_Recv(&start_time, 1, MPI_UNSIGNED_LONG, nodeID, TASK_RETURN_MATADATA_NAME, MPI_COMM_WORLD, &recv_status);

        unsigned long end_time = std::time(0);
        outfile << end_time << ",Complete_Map Task," << chunkID << ","
                << end_time - start_time << "\n";

        /* record reduce filename */
        for (int i = 0; i < info[0]; ++i) {
            if (reducer_files.count(reducerIDs[i]) == 0) {
                std::vector<int> chunkIDs;
                reducer_files[reducerIDs[i]] = chunkIDs;
            }
            reducer_files[reducerIDs[i]].push_back(info[1]);
        }
        ++recv_chunk_num;
    }

    unsigned long start_time;
    start_time = std::time(0);
    outfile << start_time << ",Start_Shuffle," << num_K_V_pair <<"\n";
    for (int reduceID = 1; reduceID <= num_reducer; ++reduceID) {
        std::ofstream out(metadata_dir + job_name + "-" + std::to_string(reduceID) +
                           "-metadata.out");

        for (auto chunkID : reducer_files[reduceID]) {
            std::ifstream in(metadata_dir + std::to_string(reduceID) +
                             "-" + std::to_string(chunkID) + "-metadata.out");
            
            std::string line;
            while (getline(in, line)) {
                out << line << "\n";
            }
            in.close();
        }

        out.close();
    }
    unsigned long end_time = std::time(0);
    outfile << end_time << ",Finish_Shuffle," << end_time - start_time << "\n";

    pthread_exit(NULL);
}

void *recv_reduce_complete(void *opaque) {
    /* set cpu core */
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(1, &cpu_set);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);

    std::ofstream &outfile = *(std::ofstream *)opaque;

    int complete_reduce_task = 0;
    MPI_Status recv_status;
    int reduceID;
    while (complete_reduce_task < num_reducer) {
        MPI_Recv(&reduceID, 1, MPI_INT, MPI_ANY_SOURCE, TASK_RETURN_COMPELTE,
                 MPI_COMM_WORLD, &recv_status);

        int nodeID = recv_status.MPI_SOURCE;
        unsigned long start_time;
        MPI_Recv(&start_time, 1, MPI_UNSIGNED_LONG, nodeID, TASK_RETURN_COMPELTE,
                 MPI_COMM_WORLD, &recv_status);

        unsigned long end_time = std::time(0);
        outfile << end_time << ",Complete_ReduceTask," << reduceID << ","
                << end_time - start_time << "\n";
        ++complete_reduce_task;
    }
    pthread_exit(NULL);
}

void jobtracker() {
    /* set cpu core */
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(0, &cpu_set);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);

    std::ofstream outfile(output_dir + "/" + job_name + "-log.out");

    unsigned long job_start_time = std::time(0);

    outfile << job_start_time << ",Start_Job," << job_name << ","
            << num_nodes << "," << ncpus << "," << num_reducer << ","
            << delay << "," << input_filename << "," << chunk_size << ","
            << locality_config_filename << "," << output_dir << std::endl;


    /* read locality config file */
    std::map<int, int> locality_config;
    read_locality_config_file(locality_config);

    /* create receive reduce filename thread */
    pthread_t recv_reduce_filename_thread;
    num_chunk = locality_config.size();
    pthread_create(&recv_reduce_filename_thread, NULL, recv_reduce_filename, (void *)&outfile);

    /* contact with mapper */
    int stop_mappers = 0;
    while (stop_mappers < (num_nodes - 1) * (ncpus - 1)) {
        int nodeID, chunkID, mapperID;
        MPI_Status recv_status;
        MPI_Recv(&mapperID, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD,
                 &recv_status);

        nodeID = recv_status.MPI_SOURCE;
        chunkID = get_chunk_by_locality_config(locality_config, nodeID);

        /* send chunk ID */
        unsigned long start_time = std::time(0);
        MPI_Send(&chunkID, 1, MPI_INT, nodeID, mapperID + MAX_MPI_TAG, MPI_COMM_WORLD);
        MPI_Send(&start_time, 1, MPI_UNSIGNED_LONG, nodeID, mapperID + MAX_MPI_TAG, MPI_COMM_WORLD);

        if (chunkID == 0) {
            ++stop_mappers;
            continue;
        }
        outfile << start_time << ",Dispatch_Map Task," << abs(chunkID) << "," << mapperID << "\n";
    }

    pthread_join(recv_reduce_filename_thread, NULL);

    /* create receive reduce task complete thread */
    pthread_t recv_reduce_complete_thread;
    pthread_create(&recv_reduce_complete_thread, NULL, recv_reduce_complete, (void *)&outfile);

    /* contact with reducer */
    int stop_reducers = 0;
    int reduceID = 1;
    while (stop_reducers < num_nodes - 1) {
        int reducerID;
        MPI_Status recv_status;
        MPI_Recv(&reducerID, 1, MPI_INT, MPI_ANY_SOURCE, TASK_REQUEST_REDUCE_ID, MPI_COMM_WORLD,
                 &recv_status);

        int data;
        if (reduceID <= num_reducer) {
            data = reduceID;
        } else {
            data = 0;
            ++stop_reducers;
        }
        
        unsigned long start_time = std::time(0);
        MPI_Send(&data, 1, MPI_INT, reducerID, JOB_RETURN_REDUCE_ID, MPI_COMM_WORLD);
        MPI_Send(&start_time, 1, MPI_UNSIGNED_LONG, reducerID, JOB_RETURN_REDUCE_ID, MPI_COMM_WORLD);
        
        if (data != 0) {
            outfile << start_time << ",Dispatch_Reduce Task," << reduceID
                    << "," << reducerID << "\n";
        }
        
        ++reduceID;
    }

    pthread_join(recv_reduce_complete_thread, NULL);

    unsigned long job_end_time = std::time(0);
    outfile << job_end_time << ",Finish_Job,"
            << job_end_time - job_start_time << "\n";

    outfile.close();
}

int main(int argc, char **argv)
{
    assert(argc == 8);
    job_name = std::string(argv[1]);
    num_reducer = std::stoi(argv[2]);
    delay = std::stoi(argv[3]);
    input_filename = std::string(argv[4]);
    chunk_size = std::stoi(argv[5]);
    locality_config_filename = std::string(argv[6]);
    output_dir = std::string(argv[7]);

    /* MPI init */
    int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE , &provided);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // the total number of process
	MPI_Comm_size(MPI_COMM_WORLD, &num_nodes); // the rank (id) of the calling process

    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);

    pthread_t jobtracker_thread;
    if (rank == 0) {
        jobtracker();
    } else {
        tasktracker();
    }
    
    MPI_Finalize();
    return 0;
}