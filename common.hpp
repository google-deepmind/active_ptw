#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cassert>
#include <limits>
#include <utility>


/* -------------------------------------------------------------------------- */


// sufficient statistics for Beta distribution
using beta_suff_stats_t = std::pair<double, double>;


/* -------------------------------------------------------------------------- */


/* given log(x) and log(y), compute log(x+y). uses the following identity:
   log(x + y) = log(x) + log(1 + y/x) = log(x) + log(1+exp(log(y)-log(x)))*/
inline double logAdd(double log_x, double log_y) {

    // ensure log_y >= log_x, can save some expensive log/exp calls
    if (log_x > log_y) {
        double t = log_x; log_x = log_y; log_y = t;
    }

    double rval = log_y - log_x;

    // only replace log(1+exp(log(y)-log(x))) with log(y)-log(x)
    // if the the difference is small enough to be meaningful
    if (rval < 100.0) rval = std::log1p(std::exp(rval));

    rval += log_x;
    return rval;
}


/* -------------------------------------------------------------------------- */


/* Bernoulli relative entropy between B(p) and B(q), handling the edge cases. */
inline double bernoulliRelEntropy(double p, double q) {

    if (p < 0.0 || q < 0.0 || p > 1.0 || q > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // handle singularities
    if (p == 0.0 && q == 0.0) {
        return 0.0;
    } else if (p == 1.0 && q == 1.0) {
        return 0.0;
    }

    if (p == 0.0) {
        return -std::log(1.0 - q);
    } else if (p == 1.0) {
        return -std::log(q);
    }

    if (q == 0.0 || q == 1.0) {
        return std::numeric_limits<double>::infinity();
    }

    // otherwise
    return p * std::log(p/q) + (1.0-p) * std::log((1.0 - p)/(1.0 - q));
}


/* -------------------------------------------------------------------------- */


/* Print an error message and exit the program. */
inline void die_with_error(const char *errmsg) {

    std::cerr << errmsg << std::endl;
    std::exit(1);
}


/* -------------------------------------------------------------------------- */



#endif // __COMMON_HPP__
