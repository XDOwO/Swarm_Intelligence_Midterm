#include <random>
#include <vector>
#include <cmath>

using gene_t = std::vector<double>;
using matrix_t = std::vector<gene_t>;

inline double dot(const gene_t &a, const gene_t &b) {
    double sm = 0;
    for (int i = 0; i < a.size(); ++i) {
        sm += a[i] * b[i];
    }
    return sm;
}

inline void normalize(gene_t &v) {
    double norm = sqrt(dot(v, v));
    if (norm > 1e-12) {
        for (auto &x : v)
            x /= norm;
    }
}

inline void subtract_projection(gene_t &v, const gene_t &u) {
    double scale = dot(v, u);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] -= scale * u[i];
}

inline matrix_t orthogonalize(const matrix_t &basis, int dim) {
    matrix_t ortho;
    for (const auto &vec : basis) {
        gene_t v = vec;
        for (const auto &u : ortho)
            subtract_projection(v, u);
        if (sqrt(dot(v, v)) > 1e-6) {
            normalize(v);
            ortho.push_back(v);
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    while ((int)ortho.size() < dim) {
        gene_t v(dim);
        for (auto &x : v)
            x = dis(gen);
        for (const auto &u : ortho)
            subtract_projection(v, u);
        if (sqrt(dot(v, v)) > 1e-6) {
            normalize(v);
            ortho.push_back(v);
        }
    }
    return ortho;
}

inline matrix_t transpose(const matrix_t &M) {
    size_t rows = M.size(), cols = M[0].size();
    matrix_t T(cols, gene_t(rows));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            T[j][i] = M[i][j];
    return T;
}

inline gene_t matvec(const matrix_t &M, const gene_t &v) {
    gene_t result(M.size(), 0.0);
    for (size_t i = 0; i < M.size(); ++i)
        for (size_t j = 0; j < v.size(); ++j)
            result[i] += M[j][i] * v[j];
    return result;
}

inline matrix_t matmat(const matrix_t &A, const matrix_t &B) {
    size_t m = A.size(), n = B.size(), p = B[0].size();
    matrix_t result(m, gene_t(p, 0.0));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < p; ++j)
            for (size_t k = 0; k < n; ++k)
                result[j][i] += A[k][i] * B[j][k];
    return result;
}