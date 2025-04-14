
#include "../test_function.h"
#include <algorithm>
#include <cmath>
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

class LSHADE {
  public:
    int dim, func_num, eval_amt, tot_eval;
    int pop_size, init_pop_size, min_pop_size, arc_size, memory_size;
    double lower_bound, upper_bound;
    vector<double> MCR, MF;
    int memory_pos = 0;

    random_device rd;
    mt19937_64 gen;
    uniform_real_distribution<> dis;
    uniform_real_distribution<> dis_range;

    struct individual {
        gene_t genes;
        double fitness;
    };

    LSHADE(int d, int fnum) : dim(d), func_num(fnum), gen(rd()), dis(0.0, 1.0) {
        set_search_bound(&upper_bound, &lower_bound, func_num);
        dis_range = uniform_real_distribution<>(lower_bound, upper_bound);
        eval_amt = 10000 * dim;
        tot_eval = eval_amt;
        pop_size = int(18 * dim);
        init_pop_size = pop_size;
        min_pop_size = 4;
        arc_size = pop_size * 1.4;
        memory_size = 5;
        MCR.assign(memory_size, 0.5);
        MF.assign(memory_size, 0.5);
    }

    double bound(double val, double parent) {
        if (val < lower_bound) return (lower_bound + parent) / 2.0;
        if (val > upper_bound) return (upper_bound + parent) / 2.0;
        return val;
    }

    double evaluate(const gene_t &x) {
        --eval_amt;
        return calculate_test_function(x.data(), dim, func_num);
    }

    double apply() {
        vector<individual> pop(pop_size);
        for (auto &ind : pop) {
            ind.genes.resize(dim);
            for (auto &g : ind.genes) g = dis_range(gen);
            ind.fitness = evaluate(ind.genes);
        }
        vector<individual> archive;
        individual best = *min_element(pop.begin(), pop.end(), [](const auto &a, const auto &b) {
            return a.fitness < b.fitness;
        });

        while (eval_amt > 0 && pop_size > min_pop_size) {
            vector<double> SCR, SF, df;
            vector<individual> new_pop;
            vector<int> sorted_idx(pop_size);
            iota(sorted_idx.begin(), sorted_idx.end(), 0);
            sort(sorted_idx.begin(), sorted_idx.end(), [&](int i, int j) {
                return pop[i].fitness < pop[j].fitness;
            });

            int p_num = max(2, int(round(pop_size * 0.1)));
            for (int i = 0; i < pop_size && eval_amt > 0; ++i) {
                int r = uniform_int_distribution<int>(0, memory_size - 1)(gen);
                double mu_cr = MCR[r];
                double mu_f = MF[r];

                normal_distribution<double> cr_dist(mu_cr, 0.1);
                double CR = clamp(cr_dist(gen), 0.0, 1.0);

                double F;
                do {
                    F = mu_f + 0.1 * tan(M_PI * (dis(gen) - 0.5));
                } while (F <= 0);
                F = min(F, 1.0);

                int pbest = sorted_idx[uniform_int_distribution<int>(0, p_num - 1)(gen)];
                int a, b;
                do { a = uniform_int_distribution<int>(0, pop_size - 1)(gen); } while (a == i);
                do { b = uniform_int_distribution<int>(0, pop_size + archive.size() - 1)(gen); } while (b == i || b == a);

                gene_t &x = pop[i].genes;
                gene_t &xp = pop[pbest].genes;
                gene_t &xa = pop[a].genes;
                gene_t &xb = (b < pop_size) ? pop[b].genes : archive[b - pop_size].genes;

                gene_t v(dim);
                for (int j = 0; j < dim; ++j)
                    v[j] = bound(x[j] + F * (xp[j] - x[j]) + F * (xa[j] - xb[j]), x[j]);

                gene_t u = x;
                int R = uniform_int_distribution<int>(0, dim - 1)(gen);
                for (int j = 0; j < dim; ++j)
                    if (dis(gen) < CR || j == R) u[j] = v[j];

                double fit_u = evaluate(u);
                if (fit_u < pop[i].fitness) {
                    new_pop.push_back({u, fit_u});
                    archive.push_back(pop[i]);
                    SCR.push_back(CR);
                    SF.push_back(F);
                    df.push_back(pop[i].fitness - fit_u);
                    if (fit_u < best.fitness) best = {u, fit_u};
                } else {
                    new_pop.push_back(pop[i]);
                }
            }

            if (!SF.empty()) {
                double sum_df = accumulate(df.begin(), df.end(), 0.0);
                double mf = 0, mcr = 0, fw = 0, wsum = 0;
                for (size_t k = 0; k < SF.size(); ++k) {
                    double w = df[k] / sum_df;
                    wsum += w;
                    mf += w * SF[k] * SF[k];
                    fw += w * SF[k];
                    mcr += w * SCR[k];
                }
                MF[memory_pos] = mf / (fw + 1e-12);
                MCR[memory_pos] = mcr / (wsum + 1e-12);
                memory_pos = (memory_pos + 1) % memory_size;
            }

            int next_size =
                int(init_pop_size - (init_pop_size - min_pop_size) *
                                   (double)(tot_eval - eval_amt) / tot_eval);
            if ((int)new_pop.size() > next_size) {
                sort(new_pop.begin(), new_pop.end(), [](auto &a, auto &b) {
                    return a.fitness < b.fitness;
                });
                new_pop.resize(next_size);
            }
            pop = new_pop;
            pop_size = pop.size();
            arc_size = pop_size * 1.4;

            if ((int)archive.size() > arc_size) {
                shuffle(archive.begin(), archive.end(), gen);
                archive.resize(arc_size);
            }
        }
        return best.fitness;
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
    for (int i = 0; i < times ; ++i) {
        LSHADE LSHADE_(dim, func_num);
        double res = LSHADE_.apply();
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

