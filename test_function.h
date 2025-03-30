/* --- Test function for 112 midterm report of NSYSU swarm intelligence --- */
// Editor: Cheng-Chun Chen

#include <math.h>
#include <float.h>
#include <stdio.h>

void set_search_bound(double *upper_bound, double *lower_bound, const int func_num);
double calculate_test_function(const double *x, const int d, const int func_num);

double Ackley(const double *x, const int d);
double Rastrigin(const double *x, const int d);
double HappyCat(const double *x, const int d);
double Rosenbrock(const double *x, const int d);
double Zakharov(const double *x, const int d);
double Michalewicz(const double *x, const int d);

void set_search_bound(double *upper_bound, double *lower_bound, const int func_num)
{
    switch (func_num)
    {
    case 1:
        *upper_bound = 32.768;
        *lower_bound = -32.768;
        break;
    case 2:
        *upper_bound = 5.12;
        *lower_bound = -5.12;
        break;

    case 3:
        *upper_bound = 20.0;
        *lower_bound = -20.0;
        break;

    case 4:
        *upper_bound = 10.0;
        *lower_bound = -10.0;
        break;

    case 5:
        *upper_bound = 10.0;
        *lower_bound = -10.0;
        break;

    case 6:
        *upper_bound = M_PI;
        *lower_bound = 0.0;
        break;

    default:
        printf("Invalid function number");
        break;
    }
}

double calculate_test_function(const double *x, const int d, const int func_num)
{
    double f = DBL_MAX;
    switch (func_num)
    {
    case 1:
        f = Ackley(x, d);
        break;
    case 2:
        f = Rastrigin(x, d);
        break;

    case 3:
        f = HappyCat(x, d);
        break;

    case 4:
        f = Rosenbrock(x, d);
        break;

    case 5:
        f = Zakharov(x, d);
        break;

    case 6:
        f = Michalewicz(x, d);
        break;
    default:
        printf("Invalid function number");
        break;
    }
    return f;
}

double Ackley(const double *x, const int d)
{
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < d; ++i)
    {
        sum1 += x[i] * x[i];
        sum2 += cos(2.0 * M_PI * x[i]);
    }
    return (-20.0) * exp(-0.2 * sqrt(sum1 / double(d))) - exp(sum2 / double(d)) + 20.0 + exp(1.0);
}

double Rastrigin(const double *x, const int d)
{
    double sum1 = 0.0;
    for (int i = 0; i < d; ++i)
    {
        sum1 += (x[i] * x[i]) - (10.0 * cos(2.0 * M_PI * x[i]));
    }
    return sum1 + 10.0 * double(d);
}

double HappyCat(const double *x, const int d)
{
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < d; ++i)
    {
        sum1 += x[i] * x[i];
        sum2 += x[i];
    }
    return pow(fabs(sum1 - double(d)), 0.25) + (0.5 * sum1 + sum2) / double(d) + 0.5;
}

double Rosenbrock(const double *x, const int d)
{
    double sum1 = 0.0;
    for (int i = 0; i < d - 1; ++i)
    {
        sum1 += 100.0 * (x[i + 1] - (x[i] * x[i])) * (x[i + 1] - (x[i] * x[i])) + ((x[i] - 1.0) * (x[i] - 1.0));
    }
    return sum1;
}

double Zakharov(const double *x, const int d)
{
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < d; ++i)
    {
        sum1 += x[i] * x[i];
        sum2 += 0.5 * (i + 1) * x[i];
    }
    return sum1 + pow(sum2, 2) + pow(sum2, 4);
}

double Michalewicz(const double *x, const int d)
{
    double sum1 = 0.0;
    for (int i = 0; i < d; ++i)
    {
        sum1 += sin(x[i]) * pow(sin((double(i + 1) * x[i] * x[i]) / M_PI), 20.0);
    }
    return sum1 * (-1);
}