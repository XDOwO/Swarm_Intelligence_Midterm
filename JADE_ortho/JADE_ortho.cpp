#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../matrix_utility.h"
#include "../test_function.h"

using namespace std;
using gene_t = vector<double>;
using matrix_t = vector<gene_t>;

class JADE {
  public:
    double minF;
    double maxF;
    int dim;
    int pop_size;
    int eval_amt;
    int tot_amt;
    int func_num;
    double P;
    double CRm;
    double Fm;
    double CRtau;
    double Ftau;
    double lower_bound;
    double upper_bound;
    random_device rd;
    mt19937_64 gen;
    uniform_real_distribution<> dis;
    uniform_real_distribution<> dis_range;
    normal_distribution<double> norm_cr;
    cauchy_distribution<double> norm_f;
    matrix_t basis;

    struct individual {
        gene_t genes;
        double fitness;
    };
    JADE(int d, int func_num_) {
        pop_size = 50;
        CRm = 0.5;
        Fm = 0.5;
        P = 0.1;
        CRtau = 0.1;
        Ftau = 0.1;
        gen = mt19937_64(rd());
        dis = uniform_real_distribution<>(0.0, 1.0);
        dim = d;
        func_num = func_num_;
        set_search_bound(&upper_bound, &lower_bound, func_num);
        minF = 0.0001;
        maxF = 1;
        dis_range = uniform_real_distribution<>(lower_bound, upper_bound);
        eval_amt = 10000 * dim;
        tot_amt = eval_amt;
        basis = vector<gene_t>(d, gene_t(d, 0));
        for (int i = 0; i < d; ++i) {
            basis[i][i] = 1;
        }
    }
    double bound(double val) { return max(min(upper_bound, val), lower_bound); }
    double evaluate(const gene_t &x) {
        double fitness =
            calculate_test_function(matvec(basis, x).data(), dim, func_num);
        --eval_amt;
        return fitness;
    }
    vector<individual> select_k_unique(const vector<individual> &population,
                                       int k) {
        vector<individual> result;
        random_device rd;
        mt19937 gen(rd());

        sample(population.begin(), population.end(), back_inserter(result), k,
               gen);
        return result;
    }

    double apply() {
        vector<individual> population(pop_size);
        vector<individual> archive;
        vector<individual> bests;
        individual best_one;
        int gap = 0;
        int gap_threhold = 1000;
        int generation = 1;
        double best_fitness = numeric_limits<double>::max();
        for (auto &ind : population) {
            ind.genes.resize(dim);
            for (auto &gene : ind.genes) {
                gene = dis_range(gen);
            }
        }
        for (auto &ind : population) {
            // ind.genes = matvec(transpose(basis), ind.genes);
            ind.fitness = evaluate(ind.genes);
            best_fitness = min(ind.fitness, best_fitness);
            if (best_fitness == ind.fitness) {
                best_one = ind;
            }
        }
        vector<double> CRv;
        vector<double> Fv;
        while (eval_amt) {
            vector<individual> new_population;
            vector<int> nums(pop_size, 0);
            iota(nums.begin(), nums.end(), 0);
            ranges::sort(nums, [&](int i, int j) {
                return population[i].fitness <
                       population[j].fitness; // sort by fitness
            });
            bool ChangeBasisFlag = true;
            for (int i = 0; i < pop_size && eval_amt > 0; ++i) {

                gene_t mutant(dim);
                norm_cr = normal_distribution<double>(CRm, 0.1);
                norm_f = cauchy_distribution<double>(Fm, 0.1);
                double CR = min(1.0, max(0.0, norm_cr(gen)));
                double F = min(maxF, max(minF, norm_f(gen)));
                int pidx = nums[uniform_int_distribution<int>(
                    0, max(1, int(P * pop_size)))(gen)];

                individual &pi = population[i];
                individual &pbest = population[pidx];

                int a = i;
                int b = i;
                while (a == i) {
                    a = uniform_int_distribution<int>(0, pop_size - 1)(gen);
                }
                while (b == i || b == a) {
                    b = uniform_int_distribution<int>(
                        0, pop_size - 1 + archive.size())(gen);
                }

                individual &pa = population[a];
                individual &pb =
                    b < pop_size ? population[b] : archive[b - pop_size];

                for (int j = 0; j < dim; j++) {
                    mutant[j] = pi.genes[j] +
                                F * (pbest.genes[j] - pi.genes[j]) +
                                F * (pa.genes[j] - pb.genes[j]);
                }

                mutant = std::move(matvec(basis, mutant));
                for (auto &x : mutant) {
                    x = bound(x);
                }
                mutant = std::move(matvec(transpose(basis), mutant));

                gene_t trial = pi.genes;
                int R = uniform_int_distribution<int>(0, dim - 1)(gen);
                for (int j = 0; j < dim; ++j) {
                    if (dis(gen) < CR || j == R) {
                        trial[j] = mutant[j];
                    }
                }

                double fitness = evaluate(trial);
                ++gap;
                if (fitness < population[i].fitness) {
                    new_population.push_back({trial, fitness});
                    archive.push_back(population[i]);
                    if (fitness < best_fitness) {
                        best_fitness = fitness;
                        best_one = {trial, fitness};
                        bests.push_back(best_one);
                        gap = 0;
                    }
                    CRv.push_back(CR);
                    Fv.push_back(F);
                } else {
                    new_population.push_back(population[i]);
                }
            }
            if (generation % int(tot_amt / pop_size * 0.75) == 0) {
                matrix_t new_basis(dim);
                for (int i = 0; i < dim; ++i) {
                    new_basis[i] = bests[bests.size()-1-i].genes;
                }
                new_basis = orthogonalize(new_basis, dim);
                for (auto &v : new_population) {
                    v.genes = matvec(transpose(new_basis),
                                        matvec(basis, v.genes));
                }
                for (auto &v : archive) {
                    v.genes = matvec(transpose(new_basis),
                                        matvec(basis, v.genes));
                }
                basis = std::move(new_basis);
                gap = 0;
                ChangeBasisFlag = true;
            }
            if (!CRv.empty()) {
                double Fsum = 0, F2sum = 0;
                for (auto &f : Fv) {
                    Fsum += f;
                    F2sum += f * f;
                }
                Fm = (1 - Ftau) * Fm + Ftau * (F2sum / Fsum);

                double CRsum = accumulate(CRv.begin(), CRv.end(), 0.0);
                CRm = (1 - CRtau) * CRm + CRtau * (CRsum / CRv.size());
            }
            if (archive.size() > pop_size) {
                archive.erase(archive.begin(),
                              archive.begin() + (archive.size() - pop_size));
            }
            population = new_population;
            ++generation;
        }
        return best_fitness;
    }
};

mutex io_mutex;

void run_task(int func_num, int dim, int times, const string &func_name) {
    double sm = 0;
    double mn = numeric_limits<double>::max();
    string filename = "./" + func_name + "_" + to_string(dim) + "D.txt";
    ofstream f(filename);
    if (!f) {
        lock_guard<mutex> lock(io_mutex);
        cerr << "Cannot open file: " << filename << endl;
        return;
    }
    {
        lock_guard<mutex> lock(io_mutex);
        cout << func_name << " with " << dim << " has started." << endl;
    }
    for (int i = 0; i < times; ++i) {
        JADE JADE_(dim, func_num);
        double res = JADE_.apply();
        {
            lock_guard<mutex> lock(io_mutex);
            f << res << endl;
        }
        sm += res;
        mn = min(res, mn);
    }
    f << "Avg:" << sm / 30 << endl;
    f << "Min:" << mn << endl;
    f.close();
    lock_guard<mutex> lock(io_mutex);
    cout << "Fitness Function " << func_name << " with dimension " << dim
         << " has average fitness " << sm / times << " and has minimum fitness "
         << mn << " after " << times << " runs" << endl;
}

int main(int argc, char **argv) {
    vector<int> dims = {2, 10, 30};
    vector<string> func_names = {"Ackley",     "Rastrigin", "HappyCat",
                                 "Rosenbrock", "Zakharov",  "Michalewicz"};
    vector<thread> threads;

    for (auto &dim : dims) {
        for (int func_num = 1; func_num <= 6; ++func_num) {
            threads.emplace_back(run_task, func_num, dim, 30,
                                 func_names[func_num - 1]);
        }
    }

    // threads.emplace_back(run_task, 4, 30, 30, func_names[3]);

    for (auto &t : threads) {
        t.join();
    }

    return 0;
}
