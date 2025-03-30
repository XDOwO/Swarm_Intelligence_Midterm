#include "test_function.h"
#include <random>
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>
#include <string>
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
    }
    double bound(double val){
        return max(min(upper_bound,val),lower_bound);
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
        uniform_int_distribution<int> pick(0, 2);
        for(int i=0;i < dim; ++i){
            candidates[0][i] = bound(0.5 * (p1[i]+p2[i]));
            candidates[1][i] = bound(1.5 * p1[i] - 0.5*p2[i]);
            candidates[2][i] = bound(1.5 * p2[i] - 0.5*p1[i]); 
        }
        return candidates[pick(gen)];
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
        uniform_int_distribution<int> pick(0,pop_size-1);
        while(eval_amt){
            vector<individual> new_population;
            while(new_population.size() < pop_size && eval_amt){
                const individual& p1 = population[pick(gen)];
                const individual& p2 = population[pick(gen)];

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

int main(int argc, char** argv){
    vector<int> dims={2,10,30};
    vector<string> func_names = {"Ackley","Rastrigin","HappyCat","Rosenbrock","Zakharov","Michalewicz"};
    for(int func_num=1;func_num<=6;++func_num){
        for(auto& dim:dims){
            double sm = 0;
            for(int i=0;i<30;++i){
                GA_real GA = GA_real(dim,func_num);
                double res = GA.apply();
                cout << res << endl;
                sm += res;
            }
            cout << "Fitness Function " << func_names[func_num-1] << " with dimension " << dim << " has average fitness " << sm/30 << " after 30 runs" << endl;
        }
    }
}