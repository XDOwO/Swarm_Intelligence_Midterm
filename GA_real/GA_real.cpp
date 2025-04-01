#include "../test_function.h"
#include <random>
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>
#include <string>
#include <fstream>
#include <thread>
#include <mutex>

using namespace std;
using gene_t = vector<double>;
class GA_real{
public:
    int dim;
    int pop_size;
    int eval_amt;
    int func_num;
    double crossover_rate;
    double mutation_rate;
    double mutation_range;
    double lower_bound;
    double upper_bound;
    random_device rd;
    mt19937_64 gen;
    uniform_real_distribution<> dis;
    uniform_real_distribution<> dis_range;
    uniform_int_distribution<int> pick;

    struct individual{
        gene_t genes;
        double fitness;
    };
    GA_real(int d,int func_num_){
        pop_size = 50;
        crossover_rate = 0.9;
        mutation_rate = 0.2;
        mutation_range = 0.5;
        gen = mt19937_64(rd());
        dis = uniform_real_distribution<>(0.0, 1.0);
        dim = d;
        func_num = func_num_;
        set_search_bound(&upper_bound,&lower_bound,func_num);
        dis_range = uniform_real_distribution<>(lower_bound,upper_bound);
        eval_amt = 10000 * dim;
        pick = uniform_int_distribution<int>(0,pop_size-1);
    }
    GA_real(int pop_size_,double crossover_rate_,double mutation_rate_,double mutation_range_,int func_num_,int d){
        pop_size = pop_size_;
        crossover_rate = crossover_rate_;
        mutation_rate = mutation_rate_;
        mutation_range = mutation_range_;
        gen = mt19937_64(rd());
        dis = uniform_real_distribution<>(0.0, 1.0);
        func_num = func_num_;
        dim = d;
        set_search_bound(&upper_bound,&lower_bound,func_num);
        dis_range = uniform_real_distribution<>(lower_bound,upper_bound);
        eval_amt = 10000 * dim;
        pick = uniform_int_distribution<int>(0,pop_size-1);
    }
    double bound(double val){
        return max(min(upper_bound,val),lower_bound);
    }
    int random_pick(){
        return pick(gen);
    }
    int tournament_selection(const vector<individual>& pops, int k){
        int best_id = 0;
        double best_fitness = numeric_limits<double>::max();
        for(int i=0;i<k;++i){
            int id = random_pick();
            if(pops[id].fitness < best_fitness){
                best_id = id;
                best_fitness = pops[id].fitness;
            }
        }
        return best_id;
    }
    gene_t real_crossover(const gene_t& p1,const gene_t& p2){
        gene_t child(dim);
        for(int i=0;i < dim; ++i){
            child[i] = dis(gen) < 0.5 ? p1[i] : p2[i];
        }
        return child;
    }
    gene_t linear_crossover(const gene_t& p1,const gene_t& p2){
        vector<gene_t> candidates(3,gene_t(dim));
        uniform_int_distribution<int> picks(0, 2);
        for(int i=0;i < dim; ++i){
            candidates[0][i] = bound(0.5 * (p1[i]+p2[i]));
            candidates[1][i] = bound(1.5 * p1[i] - 0.5*p2[i]);
            candidates[2][i] = bound(1.5 * p2[i] - 0.5*p1[i]); 
        }
        return candidates[picks(gen)];
    }
    void mutate(gene_t& genes){
        for(auto& gene:genes){
            if(dis(gen) < mutation_rate){
                double delta = (dis(gen)*2-1) * mutation_range;
                gene = bound(gene+delta);
            }
        }
    }
    double evaluate(const gene_t& x){
        double fitness = calculate_test_function(x.data(),dim,func_num);
        --eval_amt;
        return fitness;
    }
    double apply(){
        vector<individual> population(pop_size);
        individual best_one;
        double best_fitness = numeric_limits<double>::max();
        for(auto& ind:population){
            ind.genes.resize(dim);
            for(auto& gene:ind.genes){
                gene = dis_range(gen);
            }
            ind.fitness = evaluate(ind.genes);
            best_fitness = min(ind.fitness,best_fitness);
            if(best_fitness == ind.fitness){
                best_one = ind;
            }
        }
        while(eval_amt){
            vector<individual> new_population;
            while(new_population.size() < pop_size && eval_amt){
                const individual& p1 = population[tournament_selection(population,3)];
                const individual& p2 = population[tournament_selection(population,3)];

                individual child;

                if(dis(gen) < crossover_rate){
                    if(dis(gen) < 0.5){
                        child.genes = linear_crossover(p1.genes,p2.genes);
                    }
                    else{
                        child.genes = real_crossover(p1.genes,p2.genes);
                    }
                }
                else{
                    child.genes = p1.genes;
                }

                mutate(child.genes);
                child.fitness = evaluate(child.genes);
                if(best_fitness > child.fitness){
                    best_fitness = child.fitness;
                    best_one = child;
                }
                new_population.push_back(child);
            }
            population = new_population;
        }
        return best_fitness;
    }
};

mutex io_mutex;

void run_task(int func_num, int dim, const string& func_name) {
    double sm = 0;
    double mn = numeric_limits<double>::max();
    string filename = "./" + func_name + "_" + to_string(dim) + "D.txt";
    ofstream f(filename);
    if (!f) {
        lock_guard<mutex> lock(io_mutex);
        cerr << "Cannot open file: " << filename << endl;
        return;
    }

    for(int i = 0; i < 30; ++i) {
        GA_real GA(dim, func_num);
        double res = GA.apply();
        {
            lock_guard<mutex> lock(io_mutex);
            cout << "f" << func_num << " (" << dim << "D), run " << i+1 << ": " << res << endl;
        }
        f << res << endl;
        sm += res;
        mn = min(res,mn);
    }
    f << "Avg:" << sm / 30 << endl;
    f << "Min:" << mn << endl;
    f.close();
    lock_guard<mutex> lock(io_mutex);
    cout << "Fitness Function " << func_name << " with dimension " << dim
         << " has average fitness " << sm / 30.0 << " after 30 runs" << endl;
}

int main(int argc, char** argv){
    vector<int> dims={2,10,30};
    vector<string> func_names = {"Ackley","Rastrigin","HappyCat","Rosenbrock","Zakharov","Michalewicz"};
    vector<thread> threads;

    for (int func_num = 1; func_num <= 6; ++func_num) {
        for (auto& dim : dims) {
            threads.emplace_back(run_task, func_num, dim, func_names[func_num - 1]);
        }
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}