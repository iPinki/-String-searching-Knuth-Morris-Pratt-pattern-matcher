# include <string>
# include <vector>
#include <iostream>
#include <chrono>
#include <mpi/mpi.h>
#include <fstream>
#include <atomic>
#include <future>
#include <functional>
#include <cassert>
#include <algorithm>
#include <mutex>

using namespace std;

std::vector<std::size_t> KMP(const string& S, const string& pattern, int begin=0)
{
    vector<int> pf (pattern.length());

    pf[0] = 0;
    for (int k = 0, i = 1; i < pattern.length(); ++i)
    {
        while ((k > 0) && (pattern[i] != pattern[k]))
            k = pf[k-1];

        if (pattern[i] == pattern[k])
            k++;

        pf[i] = k;
    }

    std::vector<std::size_t> res;
    for (int k = 0, i = begin; i < S.length(); ++i)
    {
        while ((k > 0) && (pattern[k] != S[i]))
            k = pf[k-1];

        if (pattern[k] == S[i])
            k++;

        if (k == pattern.length())
            res.push_back(i - pattern.length() + 1);
    }

    return res;
}


std::pair<std::string, std::string> generateRandomTest(int pattern_size, int data_size){
    std::string pattern;
    std::string data;
    pattern.resize(pattern_size);
    data.resize(data_size);
    for (auto it = pattern.begin(); it!=pattern.end();){
        char c = rand();
        if (std::isalpha(c) && std::islower(c)){
            *it = c;
            ++it;
        }
    }
    for (auto it = data.begin(); it!=data.end();){
        char c = rand();
        if (std::isalpha(c) && std::islower(c)){
            *it = c;
            ++it;
        }
    }
    return std::make_pair(std::move(pattern), std::move(data));
}

vector<std::pair<std::string, std::string>> generateTests(int count, int pattern_size, int data_size){
    vector<std::pair<std::string, std::string>> res;
    res.reserve(count);
    for (int i=0; i<count;i++){
        res.push_back(generateRandomTest(pattern_size, data_size));
    }
    return res;
}

void single(std::ostream& file, const vector<std::pair<std::string, std::string>>& tests, vector<bool>& check){
    auto begin = std::chrono::steady_clock::now();
    file<<"Single thread:\n";
    int res_count=0;
    for (int i = 0; i < tests.size(); i++) {
        res_count+=KMP(tests[i].first, tests[i].second).size();
        check[i] = true;
    }
    file<<"patterns found:"<<res_count<<"\n";
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    file<<"time (ms): "<<elapsed_ms.count()<<"\n\n";
}

void mpi(std::ostream& file, const vector<std::pair<std::string, std::string>>& tests, int *argc, char ***argv, vector<bool>& check){
    int rank, size;
    auto begin = std::chrono::steady_clock::now();//the costs of creating threads must also be considered
    file<<"Mpi:\n";
    int res_count=0;
    MPI_Init(argc, argv);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    for (int i = rank; i<tests.size();i+=size) {
        res_count+=KMP(tests[i].first, tests[i].second).size();
        check[i] = true;
    }
    MPI_Finalize();
    file<<"patterns found:"<<res_count<<"\n";
    auto end = std::chrono::steady_clock::now();//the costs of creating threads must also be considered
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    file<<"time (ms): "<<elapsed_ms.count()<<"\n\n";
}

void openmp(std::ostream& file, const vector<std::pair<std::string, std::string>>& tests, vector<bool>& check){
    auto begin = std::chrono::steady_clock::now();
    file<<"OpenMP thread:\n";
    int res_count=0;
    int m_test = tests.size();
    {
        #pragma omp parallel
        {
        #pragma omp for
        for (int i = 0; i < m_test; i++) {
            res_count += KMP(tests[i].first, tests[i].second).size();
            check[i] = true;
        }
        }
    }
    file<<"patterns found:"<<res_count<<"\n";
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    file<<"time (ms): "<<elapsed_ms.count()<<"\n\n";
}


//choose one of define

#define TEST_SINGLE_AND_OPENMP //just compile and run
//#define TEST_MPI //compile and run: mpiexec -n 6 ./smpl_prog where 6 is a number f threads


int main(int argc, char **argv){
    std::ofstream file("res.out");
    const int Test_Size = 5000;
    auto tests = generateTests(Test_Size, 10000, 5);
    vector<bool> check(Test_Size, false);
#ifdef TEST_SINGLE_AND_OPENMP
    {
        vector<bool> boolean(Test_Size, false);
        auto test1 = std::async(std::launch::deferred, std::bind(&single, std::ref(file), tests,std::ref(boolean)));
        test1.get();
        assert(std::count(boolean.begin(), boolean.end(), true) == Test_Size);
    }

    {
        vector<bool> boolean(Test_Size, false);
        auto test3 = std::async(std::launch::deferred, std::bind(&openmp, std::ref(file), tests,std::ref(boolean)));
        test3.get();
        assert(std::count(boolean.begin(), boolean.end(), true) == Test_Size);
    }

#else
    {
        int rank, size, number;
        auto begin = std::chrono::steady_clock::now();//the costs of creating threads must also be considered
        file<<"Mpi:\n";
        int res_count=0;
        {
            MPI_Init(&argc, &argv);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            assert(size>1);
            for (int i1 = rank-1; i1 < tests.size(); i1 += size-1) {
                res_count += KMP(tests[i1].first, tests[i1].second).size();
                check[i1] = true;
                number = i1;
                MPI_Send(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
            if (rank==0){
                int k=0;
                while (k<Test_Size) {
                    for(int i=1; i<size; i++) {
                        MPI_Recv(&number, 1, MPI_INT, i, 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
                        check[number] = true;
                        k++;
                    }
                }
            }
            MPI_Finalize();
        }
        if (rank==0){
            file << "patterns found:" << res_count << "\n";
            auto end = std::chrono::steady_clock::now();//the costs of creating threads must also be considered
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
            file << "time (ms): " << elapsed_ms.count() << "\n\n";
            std::flush(std::cout);
            assert(std::count(check.begin(), check.end(), true) == Test_Size);
        }
    }
#endif


    return 0;
}
