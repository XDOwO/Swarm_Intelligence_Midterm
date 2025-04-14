
#include "../test_function.h"
#include "../matrix_utility.h"
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

using namespace std;
using gene_t = vector<double>;


class SHADE {
  public:
    int dim;
    int pop_size;
    int eval_amt;
    int tot_amt;
    int func_num;
    double P;
    double lower_bound;
    double upper_bound;
    int H_size = 10;
    vector<double> MF;
    vector<double> MCR;
    int H_idx = 0;

    random_device rd;
    mt19937_64 gen;
    uniform_real_distribution<> dis;
    uniform_real_distribution<> dis_range;
    matrix_t basis;

    struct individual {
        gene_t genes;
        double fitness;
    };

    SHADE(int d, int func_num_) {
        pop_size = 50;
        dim = d;
        func_num = func_num_;
        gen = mt19937_64(rd());
        dis = uniform_real_distribution<>(0.0, 1.0);
        set_search_bound(&upper_bound, &lower_bound, func_num);
        dis_range = uniform_real_distribution<>(lower_bound, upper_bound);
        eval_amt = 10000 * dim;
        tot_amt = eval_amt;
        P = 0.1;
        MF.assign(H_size, 0.5);
        MCR.assign(H_size, 0.5);
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

    double apply() {
        vector<individual> population(pop_size);
        vector<individual> archive;
        individual best_one;
        double best_fitness = numeric_limits<double>::max();

        vector<individual> bests;
        int gap = 0;
        int gap_threhold = 1000;
        int generation = 1;

        for (auto &ind : population) {
            ind.genes.resize(dim);
            for (auto &gene : ind.genes)
                gene = dis_range(gen);
            ind.fitness = evaluate(ind.genes);
            if (ind.fitness < best_fitness) {
                best_fitness = ind.fitness;
                best_one = ind;
            }
        }

        while (eval_amt) {
            vector<individual> new_population;
            vector<int> nums(pop_size);
            iota(nums.begin(), nums.end(), 0);
            ranges::sort(nums, [&](int i, int j) {
                return population[i].fitness < population[j].fitness;
            });

            vector<double> success_CR, success_F;
            vector<double> delta_fitness;

            for (int i = 0; i < pop_size && eval_amt > 0; ++i) {
                int r = uniform_int_distribution<int>(0, H_size - 1)(gen);
                normal_distribution<double> dist_cr(MCR[r], 0.1);
                cauchy_distribution<double> dist_f(MF[r], 0.1);

                double CR = min(1.0, max(0.0, dist_cr(gen)));
                double F = dist_f(gen);
                while (F <= 0.0) F = dist_f(gen);
                F = min(F, 1.0);

                int p_max = max(2, int(P * pop_size));
                int pidx = nums[uniform_int_distribution<int>(0, p_max - 1)(gen)];
                individual &pi = population[i];
                individual &pbest = population[pidx];

                int a, b;
                do { a = uniform_int_distribution<int>(0, pop_size - 1)(gen); } while (a == i);
                do {
                    b = uniform_int_distribution<int>(0, pop_size + archive.size() - 1)(gen);
                } while (b == i || b == a);

                individual &pa = population[a];
                individual &pb = (b < pop_size ? population[b] : archive[b - pop_size]);

                gene_t mutant(dim);
                for (int j = 0; j < dim; ++j) {
                    mutant[j] = pi.genes[j] + F * (pbest.genes[j] - pi.genes[j]) +
                                F * (pa.genes[j] - pb.genes[j]);
                    mutant[j] = bound(mutant[j]);
                }

                mutant = std::move(matvec(basis, mutant));
                for (auto &x : mutant) {
                    x = bound(x);
                }
                mutant = std::move(matvec(transpose(basis), mutant));

                gene_t trial = pi.genes;
                int R = uniform_int_distribution<int>(0, dim - 1)(gen);
                for (int j = 0; j < dim; ++j) {
                    if (dis(gen) < CR || j == R) trial[j] = mutant[j];
                }

                double trial_fitness = evaluate(trial);
                ++gap;
                if (trial_fitness <= pi.fitness) {
                    new_population.push_back({trial, trial_fitness});
                    archive.push_back(pi);
                    success_CR.push_back(CR);
                    success_F.push_back(F);
                    delta_fitness.push_back(abs(pi.fitness - trial_fitness));
                    if (trial_fitness < best_fitness) {
                        best_fitness = trial_fitness;
                        best_one = {trial, trial_fitness};
                        bests.push_back(best_one);
                        gap = 0;
                    }
                } else {
                    new_population.push_back(pi);
                }
            }
            if (generation % int(tot_amt / pop_size * 0.75) == 0) {
                matrix_t new_basis(dim);
                for (int i = 0; i < dim; ++i) {
                    new_basis[i] = bests[bests.size() - 1 - i].genes;
                }
                new_basis = orthogonalize(new_basis, dim);
                for (auto &v : new_population) {
                    v.genes =
                        matvec(transpose(new_basis), matvec(basis, v.genes));
                }
                for (auto &v : archive) {
                    v.genes = matvec(transpose(new_basis),matvec(basis, v.genes));
                }
                basis = std::move(new_basis);
            }
            // Update MF, MCR
            if (!success_F.empty()) {
                double sum_dF = accumulate(delta_fitness.begin(), delta_fitness.end(), 0.0);
                if (sum_dF > 1e-12) {
                    double mf_new = 0, mcr_new = 0, Fw_denom = 0, wsum = 0;
                    for (size_t k = 0; k < success_F.size(); ++k) {
                        double w = delta_fitness[k] / sum_dF;
                        wsum += w;
                        mf_new += w * success_F[k] * success_F[k];
                        Fw_denom += w * success_F[k];
                        mcr_new += w * success_CR[k];
                    }
                    MF[H_idx] = mf_new / (Fw_denom + 1e-12);
                    MCR[H_idx] = mcr_new / wsum;
                    H_idx = (H_idx + 1) % H_size;
                }
            }

            if (archive.size() > pop_size) {
                archive.erase(archive.begin(), archive.begin() + (archive.size() - pop_size));
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
    for (int i = 0; i < 30; ++i) {
        SHADE SHADE_(dim, func_num);
        double res = SHADE_.apply();
        f << res << endl;
        sm += res;
        mn = min(res, mn);
    }
    f << "Avg:" << sm / times << endl;
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

    for (int func_num = 1; func_num <= 6; ++func_num) {
        for (auto &dim : dims) {
            threads.emplace_back(run_task, func_num, dim, 30,
                                 func_names[func_num - 1]);
        }
    }
    // threads.emplace_back(run_task, 2, 30, 1, func_names[1]);

    for (auto &t : threads) {
        t.join();
    }

    return 0;
}

