#include "../test_function.h"
#include <random>
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>
#include <string>
#include <ranges>
#include <fstream>
#include <thread>
#include <mutex>

using namespace std;
using gene_t = vector<double>;
class DE{
public:
    double F;
    int dim;
    int pop_size;
    int eval_amt;
    int tot_amt;
    int func_num;
    double crossover_rate;
    double lower_bound;
    double upper_bound;
    random_device rd;
    mt19937_64 gen;
    uniform_real_distribution<> dis;
    uniform_real_distribution<> dis_range;

    struct individual{
        gene_t genes;
        double fitness;
    };
    DE(int d,int func_num_){
        pop_size = 50;
        crossover_rate = 0.9;
        gen = mt19937_64(rd());
        dis = uniform_real_distribution<>(0.0, 1.0);
        dim = d;
        func_num = func_num_;
        set_search_bound(&upper_bound,&lower_bound,func_num);
        F = (upper_bound-lower_bound)/100;
        dis_range = uniform_real_distribution<>(lower_bound,upper_bound);
        eval_amt = 10000 * dim;
        tot_amt = eval_amt;
    }
    DE(int pop_size_,double F_,double crossover_rate_,int func_num_,int d){
        pop_size = pop_size_;
        F = F_;
        crossover_rate = crossover_rate_;
        gen = mt19937_64(rd());
        dis = uniform_real_distribution<>(0.0, 1.0);
        func_num = func_num_;
        dim = d;
        set_search_bound(&upper_bound,&lower_bound,func_num);
        dis_range = uniform_real_distribution<>(lower_bound,upper_bound);
        eval_amt = 10000 * dim;
        tot_amt = eval_amt;
    }
    double bound(double val){
        return max(min(upper_bound,val),lower_bound);
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
            vector<int> nums(pop_size,0);
            ranges::iota(nums,0);
            for(int i=0;i<pop_size && eval_amt>0;++i){
                nums.erase(nums.begin()+i);
                ranges::shuffle(nums,gen);
                individual a = population[nums[0]]; 
                individual b = population[nums[1]]; 
                individual c = population[nums[2]]; 
                gene_t mutant(dim);
                
                if(dis(gen) < 0.5){
                    for(int j=0;j<dim;++j){
                        mutant[j] = bound(a.genes[j] + F * (b.genes[j] - c.genes[j]));
                    }
                }
                else{
                    for(int j=0;j<dim;++j){
                        mutant[j] = bound(population[i].genes[j] + F * (best_one.genes[j] - population[i].genes[j]) + F * (a.genes[j] - b.genes[j]));
                    }
                }

                gene_t trial(dim);
                int R = uniform_int_distribution<int>(0,pop_size-1)(gen);
                for(int j=0;j<dim;++j){
                    if(dis(gen) < crossover_rate || j==R){
                        trial[j] = mutant[j];
                    }
                    else{
                        trial[j] = population[i].genes[j];
                    }
                }

                double fitness = evaluate(trial);

                if(fitness < population[i].fitness){
                    new_population.push_back({trial,fitness});
                    if(fitness < best_fitness){
                        best_fitness = fitness;
                        best_one = {trial,fitness};
                    }
                }
                else{
                    new_population.push_back(population[i]);
                }
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
        DE DE_(dim, func_num);
        double res = DE_.apply();
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