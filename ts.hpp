#ifndef __TS_HPP__
#define __TS_HPP__

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "bandits.hpp"
#include "ptw.hpp"


/* -------------------------------------------------------------------------- */


class ThompsonSamplingStrategy : public BanditStrategy {

    public:

        ThompsonSamplingStrategy(unsigned int seed, size_t n_arms);

        // get the action using a thompson sampling strategy
        size_t getAction() override;

        // update the internal PTW environment statistics
        void update(size_t arm, int reward) override;

        // vanilla Thompson Sampling
        std::string name() const override { return "TS"; }

    private:

        mutable std::default_random_engine m_generator;

        // models the environment using a Beta distribution
        // that is updated using Bayesian inference
        std::vector<KTEstimator> m_model;
};


/* -------------------------------------------------------------------------- */


class ActivePTWBanditStrategy : public BanditStrategy {

    public:

        ActivePTWBanditStrategy(unsigned int seed, size_t n_arms);

        // get the action using a thompson sampling strategy
        size_t getAction() override;

        // update the internal PTW environment statistics
        void update(size_t arm, int reward) override;

        std::string name() const override { return "ActivePTW"; }

        // the posterior probability of being in a segment of length 2^k
        std::vector<double> levelPosterior() const;

        // sample according to the posterior over segments
        size_t levelPosteriorSample() const;

        // PTW statistics accessor
        const ActivePTW &model() const;

    private:

        mutable std::default_random_engine m_generator;

        ActivePTW m_model;
        size_t m_arms;
};


/* -------------------------------------------------------------------------- */


class ParanoidPTWBanditStrategy : public BanditStrategy {

    public:

        ParanoidPTWBanditStrategy(unsigned int seed, size_t n_arms);

        size_t getAction() override;

        void update(size_t arm, int reward) override;

        std::string name() const override { return "ParanoidPTW"; }

    private:

        // determine the rate of forced exploration
        // based on the segment size
        double exploreProb(size_t log2_segment_size) const;

        // given a segment at a given level, determine
        //  the least explored arm
        size_t leastExploredArm(size_t level) const;

        mutable std::default_random_engine m_generator;

        size_t m_arms;
        ActivePTWBanditStrategy m_aptw;
        size_t m_trials;
};


/* -------------------------------------------------------------------------- */


#endif // __TS_HPP__

