
#include "../test_function.h"
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
    int func_num;
    int eval_amt;
    int tot_amt;
    int pop_size;
    int min_pop_size = 4;
    int memory_size = 5;
    int arc_size;
    double lower_bound, upper_bound;
    double epsilon = 1e-8;
    double p_best_rate = 0.25;
    double arc_rate = 1.0;

    vector<double> M_F;
    vector<double> M_CR;
    int memory_pos = 0;

    struct individual {
        gene_t genes;
        double fitness;
    };

    vector<individual> population;
    vector<individual> archive;

    random_device rd;
    mt19937_64 gen;
    uniform_real_distribution<> dis;
    uniform_real_distribution<> dis_range;
    normal_distribution<double> norm_dist;

    JSO(int d, int fnum) : dim(d), func_num(fnum), gen(rd()), dis(0.0, 1.0), norm_dist(0.0, 1.0) {
        set_search_bound(&upper_bound, &lower_bound, func_num);
        dis_range = uniform_real_distribution<>(lower_bound, upper_bound);
        pop_size = int(round(sqrt(dim) * log(dim) * 25));
        arc_size = int(round(pop_size * arc_rate));
        eval_amt = 10000 * dim;
        tot_amt = eval_amt;

        M_F = vector<double>(memory_size, 0.3);
        M_CR = vector<double>(memory_size, 0.8);

        for (int i = 0; i < pop_size; ++i) {
            individual ind;
            ind.genes.resize(dim);
            for (auto &x : ind.genes)
                x = dis_range(gen);
            ind.fitness = evaluate(ind.genes);
            population.push_back(ind);
        }
    }

    double evaluate(const gene_t &x) {
        --eval_amt;
        return calculate_test_function(x.data(), dim, func_num);
    }

    gene_t bound(const gene_t &child, const gene_t &parent) {
        gene_t result = child;
        for (int i = 0; i < dim; ++i) {
            if (result[i] < lower_bound)
                result[i] = (lower_bound + parent[i]) / 2.0;
            else if (result[i] > upper_bound)
                result[i] = (upper_bound + parent[i]) / 2.0;
        }
        return result;
    }

    double apply() {
        individual best_one = *min_element(population.begin(), population.end(),
            [](const individual &a, const individual &b) { return a.fitness < b.fitness; });

        vector<double> S_F, S_CR, delta_f;

        while (eval_amt > 0 && pop_size > 2) {
            vector<individual> new_population;
            vector<int> sorted_idx(pop_size);
            iota(sorted_idx.begin(), sorted_idx.end(), 0);
            sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
                return population[a].fitness < population[b].fitness;
            });

            int p_num = max(2, min(pop_size, int(round(pop_size * p_best_rate))));

            for (int i = 0; i < pop_size && eval_amt > 0; ++i) {
                int r = uniform_int_distribution<int>(0, memory_size - 1)(gen);
                double mu_F = (r == memory_size - 1) ? 0.9 : M_F[r];
                double mu_CR = (r == memory_size - 1) ? 0.9 : M_CR[r];

                double CR = clamp(norm_dist(gen) * 0.1 + mu_CR, 0.0, 1.0);
                if (eval_amt > 0.75 * tot_amt && CR < 0.7) CR = 0.7;
                else if (eval_amt > 0.5 * tot_amt && CR < 0.6) CR = 0.6;

                double F;
                do {
                    F = mu_F +
                        0.1 * tan(3.14159265358979323846 * (dis(gen) - 0.5));
                } while (F <= 0.0);
                F = min(F, 1.0);
                if (eval_amt > 0.4 * tot_amt && F > 0.7) F = 0.7;

                int pbest = sorted_idx[uniform_int_distribution<int>(0, p_num - 1)(gen)];
                while (eval_amt > 0.5 * tot_amt && pbest == i)
                    pbest = sorted_idx[uniform_int_distribution<int>(0, p_num - 1)(gen)];

                int r1, r2;
                do { r1 = uniform_int_distribution<int>(0, pop_size - 1)(gen); } while (r1 == i);
                do {
                    r2 = uniform_int_distribution<int>(0, pop_size + archive.size() - 1)(gen);
                } while (r2 == i || r2 == r1);

                gene_t xr1 = population[r1].genes;
                gene_t xr2 = (r2 < pop_size ? population[r2].genes : archive[r2 - pop_size].genes);

                double jF = F * (eval_amt > 0.8 * tot_amt ? 0.7 : eval_amt > 0.6 * tot_amt ? 0.8 : 1.2);

                gene_t vi(dim);
                for (int j = 0; j < dim; ++j) {
                    vi[j] = population[i].genes[j] +
                            jF * (population[pbest].genes[j] - population[i].genes[j]) +
                            F * (xr1[j] - xr2[j]);
                }

                vi = bound(vi, population[i].genes);

                gene_t ui = population[i].genes;
                int R = uniform_int_distribution<int>(0, dim - 1)(gen);
                for (int j = 0; j < dim; ++j) {
                    if (dis(gen) < CR || j == R) ui[j] = vi[j];
                }

                double trial_fitness = evaluate(ui);
                if (trial_fitness - 0.0 < epsilon) trial_fitness = 0.0;

                if (trial_fitness < population[i].fitness) {
                    new_population.push_back({ui, trial_fitness});
                    archive.push_back(population[i]);
                    S_F.push_back(F);
                    S_CR.push_back(CR);
                    delta_f.push_back(abs(population[i].fitness - trial_fitness));
                    if (trial_fitness < best_one.fitness) best_one = {ui, trial_fitness};
                } else {
                    new_population.push_back(population[i]);
                }
            }

            if (!S_F.empty()) {
                double sum_dF = accumulate(delta_f.begin(), delta_f.end(), 0.0);
                double wSF = 0.0, wCR = 0.0, wsF = 0.0, wsCR = 0.0;
                for (size_t i = 0; i < S_F.size(); ++i) {
                    double w = delta_f[i] / sum_dF;
                    wSF += w * S_F[i] * S_F[i];
                    wsF += w * S_F[i];
                    wCR += w * S_CR[i] * S_CR[i];
                    wsCR += w * S_CR[i];
                }
                M_F[memory_pos] = wSF / wsF;
                M_CR[memory_pos] = wsCR == 0 ? -1 : wCR / wsCR;
                memory_pos = (memory_pos + 1) % memory_size;
                S_F.clear(); S_CR.clear(); delta_f.clear();
            }

            int planned_size = int(round(((double)(min_pop_size - pop_size) / tot_amt) * (tot_amt - eval_amt) + pop_size));
            if ((int)new_population.size() > planned_size) {
                sort(new_population.begin(), new_population.end(), [](const individual &a, const individual &b) {
                    return a.fitness < b.fitness;
                });
                new_population.resize(planned_size);
            }
            if ((int)archive.size() > arc_size) archive.resize(arc_size);
            population = new_population;
            pop_size = population.size();
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
    for (int i = 0; i < 30; ++i) {
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