#ifndef __MASTER_HPP__
#define __MASTER_HPP__

#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "bandits.hpp"
#include "ucb.hpp"


/* -------------------------------------------------------------------------- */


// An implementation of the MASTER and MALG meta-algorithms with UCB1.
//  See: https://arxiv.org/pdf/2102.05406.pdf for algorithm details.


/* -------------------------------------------------------------------------- */


class MalgUCB : public BanditStrategy {

    struct instance_t {

        instance_t(
            unsigned int seed, size_t n_arms,
            size_t start, size_t end)
        :
            alg(seed, n_arms),
            s(start),
            e(end)
        {
        }

        // segment length
        size_t length() const { return e - s + 1; }

        UCBStrategy alg;   // alg instance
        size_t s;          // alg start time
        size_t e;          // alg end time
    };

    public:

        MalgUCB(unsigned int seed, size_t n_arms, size_t depth);

        // get the action from MALG
        size_t getAction() override;

        // update the algorithms internal state after
        // pulling an arm and reeiving a reward
        void update(size_t arm, int reward) override;

        // name of the method
        std::string name() const override { return "MALG"; }

    private:

        // the average regret bound used to schedule UCB instances
        double rho(double t) const;

        // get the index to the active instance
        size_t activeInstance() const;

        mutable std::default_random_engine m_generator;
        size_t m_seed;
        size_t m_arms;
        size_t m_n;
        size_t m_tau;

        std::vector<std::unique_ptr<instance_t>> m_instances;
};


/* -------------------------------------------------------------------------- */


class MasterUCB : public BanditStrategy {

    public:

        MasterUCB(size_t arms);

        // get the action from MASTER
        size_t getAction() override;

        // update the algorithms internal state after
        // pulling an arm and reeiving a reward
        void update(size_t arm, int reward) override;

        // name of the method
        std::string name() const override { return "MASTER"; }

    private:

        double rhoHat(double x) const;
};


/* -------------------------------------------------------------------------- */


#endif // __MASTER_HPP__
