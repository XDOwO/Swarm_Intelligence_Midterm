
#include "../test_function.h"
#include "../matrix_utility.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include <thread>
#include <iostream>
#include <fstream>
#include <string>
#include <mutex>

using namespace std;
using gene_t = vector<double>;

class JSO {
public:
    int dim;
    int init_pop_size;
    int pop_size;
    int eval_amt;
    int tot_amt;
    int func_num;
    int min_pop_size;
    int memory_size;
    int arc_size;
    double lower_bound;
    double upper_bound;
    double p_best_rate;
    double arc_rate;
    double epsilon;
    matrix_t basis;

    vector<double> MF, MCR;
    int H_idx = 0;

    random_device rd;
    mt19937_64 gen;
    uniform_real_distribution<> dis;
    uniform_real_distribution<> dis_range;

    struct individual {
        gene_t genes;
        double fitness;
    };

    JSO(int d, int func_num_) {
        dim = d;
        func_num = func_num_;
        gen = mt19937_64(rd());
        dis = uniform_real_distribution<>(0.0, 1.0);
        set_search_bound(&upper_bound, &lower_bound, func_num);
        dis_range = uniform_real_distribution<>(lower_bound, upper_bound);

        pop_size = (int)round(sqrt(dim) * log(dim) * 25);
        init_pop_size = pop_size;
        min_pop_size = 4;
        memory_size = 5;
        arc_rate = 1.4;
        arc_size = (int)(pop_size) * arc_rate;
        p_best_rate = 0.4;
        eval_amt = 10000 * dim;
        tot_amt = eval_amt;

        MF.assign(memory_size, 0.3);
        MCR.assign(memory_size, 0.8);

        basis = vector<gene_t>(dim, gene_t(dim, 0));
        for (int i = 0; i < dim; ++i) basis[i][i] = 1;
    }

    double bound(double val, double parent) {
        if (val < lower_bound)
            return (val + parent) / 2.0;
        if (val > upper_bound)
            return (val + parent) / 2.0;
        return val;
    }

    double evaluate(const gene_t &x) {
        double fitness = calculate_test_function(matvec(basis, x).data(), dim, func_num);
        --eval_amt;
        return fitness;
    }

    double apply() {
        vector<individual> population(pop_size);
        vector<individual> archive;
        vector<individual> bests;
        int generation = 1;
        int gap = 0;
        for (auto &ind : population) {
            ind.genes.resize(dim);
            for (auto &gene : ind.genes)
                gene = dis_range(gen);
            ind.fitness = evaluate(ind.genes);
        }

        individual best_one = *min_element(population.begin(), population.end(),
            [](const individual &a, const individual &b) { return a.fitness < b.fitness; });

        while (eval_amt > 0 && pop_size > min_pop_size) {
            vector<int> sorted_idx(pop_size);
            iota(sorted_idx.begin(), sorted_idx.end(), 0);
            ranges::sort(sorted_idx, [&](int a, int b) {
                return population[a].fitness < population[b].fitness;
            });

            int p_num = max(2, min(pop_size, int(round(pop_size * p_best_rate))));
            vector<individual> new_population;
            vector<double> S_F, S_CR, delta_fitness;

            for (int i = 0; i < pop_size && eval_amt > 0; ++i) {
                int r = uniform_int_distribution<int>(0, memory_size - 1)(gen);
                double mu_F = r == memory_size - 1 ? 0.9 : MF[r];
                double mu_CR = r == memory_size - 1 ? 0.9 : MCR[r];

                normal_distribution<double> dist_cr(mu_CR, 0.1);
                double CR = clamp(dist_cr(gen), 0.0, 1.0);
                if (eval_amt > 0.75 * tot_amt && CR < 0.7) CR = 0.7;
                else if (eval_amt > 0.5 * tot_amt && CR < 0.6) CR = 0.6;

                double F;
                do {
                    F = mu_F + 0.1 * tan(M_PI * (dis(gen) - 0.5));
                } while (F <= 0.0);
                F = min(F, 1.0);
                if (eval_amt > 0.4 * tot_amt && F > 0.7) F = 0.7;

                int pbest = sorted_idx[uniform_int_distribution<int>(0, p_num - 1)(gen)];
                while (eval_amt > 0.5 * tot_amt && pbest == i)
                    pbest = sorted_idx[uniform_int_distribution<int>(0, p_num - 1)(gen)];

                int a, b;
                do { a = uniform_int_distribution<int>(0, pop_size - 1)(gen); } while (a == i);
                do {
                    b = uniform_int_distribution<int>(0, pop_size + archive.size() - 1)(gen);
                } while (b == i || b == a);

                gene_t &xr1 = population[a].genes;
                gene_t &xr2 = (b < pop_size ? population[b].genes : archive[b - pop_size].genes);

                double jF = F * (eval_amt > 0.8 * tot_amt ? 0.7 : eval_amt > 0.6 * tot_amt ? 0.8 : 1.2);

                gene_t vi(dim);
                for (int j = 0; j < dim; ++j) {
                    vi[j] = population[i].genes[j] +
                            jF * (population[pbest].genes[j] - population[i].genes[j]) +
                            F * (xr1[j] - xr2[j]);
                    vi[j] = bound(vi[j], population[i].genes[j]);
                }

                vi = matvec(basis, vi);
                for (auto &x : vi) x = bound(x, 0);
                vi = matvec(transpose(basis), vi);

                gene_t ui = population[i].genes;
                int R = uniform_int_distribution<int>(0, dim - 1)(gen);
                for (int j = 0; j < dim; ++j)
                    if (dis(gen) < CR || j == R) ui[j] = vi[j];

                double trial_fit = evaluate(ui);
                ++gap;
                if (trial_fit < population[i].fitness) {
                    new_population.push_back({ui, trial_fit});
                    archive.push_back(population[i]);
                    S_F.push_back(F);
                    S_CR.push_back(CR);
                    delta_fitness.push_back(abs(population[i].fitness - trial_fit));
                    if (trial_fit < best_one.fitness) {
                        best_one = {ui, trial_fit};
                        bests.push_back(best_one);
                        gap = 0;
                    }
                } else {
                    new_population.push_back(population[i]);
                }
            }

            if (!S_F.empty()) {
                double sum_df = accumulate(delta_fitness.begin(), delta_fitness.end(), 0.0);
                if (sum_df > 1e-12) {
                    double mf_new = 0, Fw_denom = 0, mcr_new = 0, wsum = 0;
                    for (size_t k = 0; k < S_F.size(); ++k) {
                        double w = delta_fitness[k] / sum_df;
                        wsum += w;
                        mf_new += w * S_F[k] * S_F[k];
                        Fw_denom += w * S_F[k];
                        mcr_new += w * S_CR[k];
                    }
                    MF[H_idx] = mf_new / (Fw_denom + 1e-12);
                    MCR[H_idx] = mcr_new / wsum;
                    H_idx = (H_idx + 1) % memory_size;
                }
            }

            int planned_size = init_pop_size - (init_pop_size - min_pop_size) *
                                              (double)(tot_amt - eval_amt) /
                                              tot_amt;
            if ((int)new_population.size() > planned_size) {
                ranges::sort(new_population, [](const individual &a, const individual &b) {
                    return a.fitness < b.fitness;
                });
                new_population.resize(planned_size);
            }

            if (generation % int(tot_amt / init_pop_size * 0.75) == 0 && bests.size() >= dim && gap > dim * 1000) {
                matrix_t new_basis(dim);
                for (int i = 0; i < dim; ++i) new_basis[i] = bests[bests.size() - 1 - i].genes;
                new_basis = orthogonalize(new_basis, dim);
                for (auto &v : new_population) {
                    v.genes = matvec(transpose(new_basis), matvec(basis, v.genes));
                }
                for (auto &v : archive) {
                    v.genes = matvec(transpose(new_basis),matvec(basis, v.genes));
                }
                basis = std::move(new_basis);
                gap = 0;
            }

            population = new_population;
            pop_size = population.size();
            arc_size = pop_size * arc_rate;

            if ((int)archive.size() > arc_size) {
                shuffle(archive.begin(), archive.end(), gen);
                archive.resize(arc_size);
            }
            ++generation;
        }

        return best_one.fitness;
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
        JSO JSO_(dim, func_num);
        double res = JSO_.apply();
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