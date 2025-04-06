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
#include <unordered_set>

using namespace std;
using gene_t = vector<double>;
using matrix_t = vector<gene_t>;

double dot(const gene_t& a,const gene_t& b){
    double sm = 0;
    for(int i=0;i<a.size();++i){
        sm += a[i]*b[i];
    }
    return sm;
}

void normalize(gene_t& v) {
    double norm = sqrt(dot(v, v));
    if (norm > 1e-12) {
        for (auto& x : v) x /= norm;
    }
}

void subtract_projection(gene_t& v, const gene_t& u) {
    double scale = dot(v, u);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] -= scale * u[i];
}

matrix_t orthogonalize(const matrix_t& basis, int dim) {
    matrix_t ortho;
    for (const auto& vec : basis) {
        gene_t v = vec;
        for (const auto& u : ortho)
            subtract_projection(v, u);
        if (sqrt(dot(v, v)) > 1e-6) {
            normalize(v);
            ortho.push_back(v);
        }
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);
    while ((int)ortho.size() < dim) {
        gene_t v(dim);
        for (auto& x : v) x = dis(gen);
        for (const auto& u : ortho)
            subtract_projection(v, u);
        if (sqrt(dot(v, v)) > 1e-6) {
            normalize(v);
            ortho.push_back(v);
        }
    }
    return ortho;
}

matrix_t transpose(const matrix_t& M) {
    size_t rows = M.size(), cols = M[0].size();
    matrix_t T(cols, gene_t(rows));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            T[j][i] = M[i][j];
    return T;
}

gene_t matvec(const matrix_t& M, const gene_t& v) {
    gene_t result(M.size(), 0.0);
    for (size_t i = 0; i < M.size(); ++i)
        for (size_t j = 0; j < v.size(); ++j)
            result[i] += M[j][i] * v[j];
    return result;
}

matrix_t matmat(const matrix_t& A, const matrix_t& B) {
    size_t m = A.size(), n = B.size(), p = B[0].size();
    matrix_t result(m, gene_t(p, 0.0));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < p; ++j)
            for (size_t k = 0; k < n; ++k)
                result[j][i] += A[k][i] * B[j][k];
    return result;
}

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
    matrix_t basis;

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
        basis = vector<gene_t>(d,gene_t(d,0));
        for(int i=0;i<d;++i){
            basis[i][i] = 1;
        }
    }
    double bound(double val){
        return max(min(upper_bound,val),lower_bound);
    }
    double evaluate(const gene_t& x){
        double fitness = calculate_test_function(matvec(basis,x).data(),dim,func_num);
        --eval_amt;
        return fitness;
    }
    vector<individual> select_k_unique(const vector<individual>& population, int k) {
        vector<individual> result;
        random_device rd;
        mt19937 gen(rd());
    
        sample(population.begin(), population.end(), back_inserter(result), k, gen);
        return result;
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
        int generation = 1;
        int gap = 0;
        int gap_threhold = 1000;
        bool hasChangedBasis = false;
        while(eval_amt){
            vector<individual> new_population;
            double ns1=0,ns2=0,nf1=0,nf2=0;
            for(int i=0;i<pop_size && eval_amt>0;++i){
                auto vec = select_k_unique(population,3);
                individual a = move(vec[0]);
                individual b = move(vec[1]);
                individual c = move(vec[2]);
                gene_t mutant(dim);
                bool isDiv = true;

                norm_cr = normal_distribution<double>(CRm, 0.1);
                norm_f = normal_distribution<double>(0.5, 0.3);
                double CR = min(1.0, max(0.0, norm_cr(gen)));
                double F = min(maxF, max(minF, norm_f(gen)));
                
                if(dis(gen) < p){ //tend to diverge
                    for(int j=0;j<dim;++j){
                        mutant[j] = bound(a.genes[j] + F * (b.genes[j] - c.genes[j]));
                        // mutant[j] = bound(a.genes[j] + F * b.genes[j]);
                    }
                    isDiv = true;
                }
                else{   //tend to converge
                    for(int j=0;j<dim;++j){
                        mutant[j] = bound(population[i].genes[j] + F * (best_one.genes[j] - population[i].genes[j]) + F * (a.genes[j] - b.genes[j]));
                        // mutant[j] = bound(population[i].genes[j] + F * (best_one.genes[j] - population[i].genes[j]));                        
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
                ++gap;
                if(fitness < population[i].fitness){
                    new_population.push_back({trial,fitness});
                    if(fitness < best_fitness){
                        best_fitness = fitness;
                        best_one = {trial,fitness};
                        gap = 0;
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

            if(gap >= gap_threhold){
                for(int i=0;i<10;++i){
                    for(auto& x:new_population[i].genes){
                        x = dis_range(gen);
                    }
                }
                gap = 0;
            }

            if(gap >= gap_threhold && generation >= tot_amt/pop_size*0.75){
                ranges::sort(new_population, {}, &individual::fitness);
                matrix_t new_basis(dim);
                new_basis[0] = best_one.genes;
                for(int i=1;i<dim;++i){
                    new_basis[i] = new_population[i-1].genes;
                }
                new_basis = orthogonalize(new_basis,dim);
                for(auto& v:new_population){
                    v.genes = matvec(transpose(new_basis),matvec(basis,v.genes));
                }
                basis = move(new_basis);
                hasChangedBasis = true;
                // p = 0.5;
                // CRm = 0.5;
                // CRv.clear();
            }
            
            ++generation;
            population = new_population;

        }
        return best_fitness;
    }
};

mutex io_mutex;

void run_task(int func_num, int dim, int times,const string& func_name) {
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
        cout <<  func_name << " with " << dim << " has started." << endl;
    }
    for(int i = 0; i < times; ++i) {
        SADE SADE_(dim, func_num);
        double res = SADE_.apply();
        {
            lock_guard<mutex> lock(io_mutex);
            f << res << endl;
        }
        sm += res;
        mn = min(res,mn);
    }
    f << "Avg:" << sm / 30 << endl;
    f << "Min:" << mn << endl;
    f.close();
    lock_guard<mutex> lock(io_mutex);
    cout << "Fitness Function " << func_name << " with dimension " << dim
         << " has average fitness " << sm / times
         << " and has minimum fitness " << mn << " after " << times << " runs" << endl;
}

int main(int argc, char** argv){
    vector<int> dims={2,10,30};
    vector<string> func_names = {"Ackley","Rastrigin","HappyCat","Rosenbrock","Zakharov","Michalewicz"};
    vector<thread> threads;

    for (auto& dim : dims) {
        for (int func_num=1;func_num<=6;++func_num) {
            threads.emplace_back(run_task, func_num, dim,30 ,func_names[func_num - 1]);
        }
    }

    // threads.emplace_back(run_task, 4, 30, 30, func_names[3]);

    for (auto& t : threads) {
        t.join();
    }

    return 0;

}