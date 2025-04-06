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
class SADE{
public:
    double minF;
    double maxF;
    int dim;
    int pop_size;
    int eval_amt;
    int tot_amt;
    int func_num;
    double CRm;
    double lower_bound;
    double upper_bound;
    random_device rd;
    mt19937_64 gen;
    uniform_real_distribution<> dis;
    uniform_real_distribution<> dis_range;
    normal_distribution<double> norm_cr;
    normal_distribution<double> norm_f;

    struct individual{
        gene_t genes;
        double fitness;
    };
    SADE(int d,int func_num_){
        pop_size = 50;
        CRm = 0.5;
        gen = mt19937_64(rd());
        dis = uniform_real_distribution<>(0.0, 1.0);
        dim = d;
        func_num = func_num_;
        set_search_bound(&upper_bound,&lower_bound,func_num);
        minF = 0.0001;
        maxF = 2;
        dis_range = uniform_real_distribution<>(lower_bound,upper_bound);
        eval_amt = 10000 * dim;
        tot_amt = eval_amt;
    }
    SADE(int pop_size_,double minF_,double maxF_,double CRm_,int func_num_,int d){
        pop_size = pop_size_;
        minF = minF_;
        maxF = maxF_;
        CRm = CRm_;
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
        double p=0.5;
        vector<double> CRv;
        while(eval_amt){
            vector<individual> new_population;
            vector<int> nums(pop_size,0);
            ranges::iota(nums,0);
            double ns1=0,ns2=0,nf1=0,nf2=0;
            for(int i=0;i<pop_size && eval_amt>0;++i){
                nums.erase(nums.begin()+i);
                ranges::shuffle(nums,gen);
                individual a = population[nums[0]]; 
                individual b = population[nums[1]]; 
                individual c = population[nums[2]]; 
                gene_t mutant(dim);
                bool isDiv = true;

                norm_cr = normal_distribution<double>(CRm, 0.1);
                norm_f = normal_distribution<double>(0.5, 0.3);
                double CR = min(1.0, max(0.0, norm_cr(gen)));
                double F = min(maxF, max(minF, norm_f(gen)));
                
                if(dis(gen) < p){ //tend to diverge
                    for(int j=0;j<dim;++j){
                        mutant[j] = bound(a.genes[j] + F * (b.genes[j] - c.genes[j]));
                    }
                    isDiv = true;
                }
                else{   //tend to converge
                    for(int j=0;j<dim;++j){
                        mutant[j] = bound(population[i].genes[j] + F * (best_one.genes[j] - population[i].genes[j]) + F * (a.genes[j] - b.genes[j]));
                    }
                    isDiv = false;
                }

                gene_t trial = population[i].genes;
                int R = uniform_int_distribution<int>(0,dim-1)(gen);
                for(int j=0;j<dim;++j){
                    if(dis(gen) < CR || j==R){
                        trial[j] = mutant[j];
                    }
                }

                double fitness = evaluate(trial);

                if(fitness < population[i].fitness){
                    new_population.push_back({trial,fitness});
                    if(fitness < best_fitness){
                        best_fitness = fitness;
                        best_one = {trial,fitness};
                    }
                    if(isDiv) ++ns1;
                    else ++ns2;
                    CRv.push_back(CR);
                }
                else{
                    new_population.push_back(population[i]);
                    if(isDiv) ++nf1;
                    else ++nf2;
                }

            }
            if((ns1*(ns2+nf2) + ns2*(ns1+nf1)) != 0)p = ns1*(ns2+nf2) / (ns1*(ns2+nf2) + ns2*(ns1+nf1));
            p = min(0.999,max(0.001,p));
            if(!CRv.empty()){
                CRm = accumulate(CRv.begin(),CRv.end(),0.0) / CRv.size();
                CRv.clear();
            }
            population = new_population;
        }
        return best_fitness;
    }
};

mutex io_mutex;

void run_task(int func_num, int dim,int times, const string& func_name) {
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
        SADE SADE_(dim, func_num);
        double res = SADE_.apply();
        {
            lock_guard<mutex> lock(io_mutex);
            cout << "f" << func_num << " (" << dim << "D), run " << i+1 << ": " << res << endl;
        }
        f << res << endl;
        sm += res;
        mn = min(res,mn);
    }
    f << "Avg:" << sm / times << endl;
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
            threads.emplace_back(run_task, func_num, dim, 30, func_names[func_num - 1]);
        }
    }
    
    // threads.emplace_back(run_task, 2, 30, 1, func_names[1]);

    for (auto& t : threads) {
        t.join();
    }

    return 0;

}