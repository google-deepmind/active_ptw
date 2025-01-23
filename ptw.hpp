#ifndef __PTW_HPP__
#define __PTW_HPP__

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common.hpp"


/* -------------------------------------------------------------------------- */


/* KT Estimator for binary memoryless sources */
class KTEstimator {

    const double KT_Alpha = 0.5;
    const double KT_Alpha2 = KT_Alpha + KT_Alpha;

    public:

        KTEstimator() :
            m_log_kt(0.0)
        {
            m_counts[0] = 0;
            m_counts[1] = 0;
        }

        // the probability of seeing a particular symbol next
        double prob(int b) {

            double num = double(m_counts[b]) + KT_Alpha;
            double den = double(m_counts[0] + m_counts[1]) + KT_Alpha2;
            return num / den;
        }

        // the logarithm of the probability of all processed bits
        double logMarginal() const { return m_log_kt; }

        // process a new bit
        void update(int b) {
            m_log_kt += std::log(prob(b));
            m_counts[b]++;
        }

        // give the sufficient statistics for the KT estimator
        // in the form of a beta distribution
        beta_suff_stats_t posterior() const {

            double alpha = KT_Alpha + double(m_counts[1]);
            double beta = KT_Alpha + double(m_counts[0]);

            return beta_suff_stats_t(alpha, beta);
        }

    private:

        double m_log_kt;
        uint64_t m_counts[2];
};


/* -------------------------------------------------------------------------- */


// Active Partition Tree Weighting
class ActivePTW {

    struct ActivePTWNode_t {

        ActivePTWNode_t(size_t arms) :
            m_model(arms),
            m_log_weighted(0.0),
            m_log_buf(0.0)
        {
        }

        // the probability of a segment is equal to the product
        // of each subsequence explained by each arm
        double logMarginal() const {

            double rval = 0.0;
            for (size_t i = 0; i < m_model.size(); i++) {
                rval += m_model[i].logMarginal();
            }

            return rval;
        }

        double prob(int r, size_t k) { return m_model[k].prob(r); }

        std::vector<KTEstimator> m_model;
        double m_log_weighted;
        double m_log_buf;
    };


    public:

        typedef uint64_t index_t;

        ActivePTW(size_t depth, size_t arms);

        // the probability of seeing a reward r next if arm k pulled
        double prob(int r, size_t k);

        // the logarithm of the probability of all processed bits
        double logMarginal() const { return m_nodes[0].m_log_weighted; }

        // process a new piece of experience, indicating arm k
        // pulled with reward r
        void update(int r, size_t k);

        // the posterior probability of being in a segment of length 2^k
        std::vector<double> levelPosterior() const;

        // given a segmentation level, and choice of arm, what is the posterior
        // probability which governs the arm's latent reward distribution
        beta_suff_stats_t posterior(size_t level, size_t arm_index) const;

    private:

        // the number of bits to the left of the most significant
        // location at which times t-1 and t-2 differ, where t is
        // the 1 based representation of the current time
        size_t mscb(index_t t) const;

        index_t m_index;
        std::vector<ActivePTWNode_t> m_nodes;
        size_t m_depth;
        size_t m_arms;

        // parameters to define the PTW prior
        double LogSplitWeight = std::log(0.5);
        double LogStopWeight  = std::log(0.5);
};


/* -------------------------------------------------------------------------- */


#endif // __PTW_HPP__


