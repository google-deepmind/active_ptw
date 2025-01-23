#ifndef __UNIFORM_HPP__
#define __UNIFORM_HPP__

#include <cstddef>
#include <string>
#include <random>

#include "bandits.hpp"


/* -------------------------------------------------------------------------- */


class UniformSamplingStrategy : public BanditStrategy {

    public:

        UniformSamplingStrategy(unsigned int seed, size_t n_arms);

        // pick an action uniformly at random
        size_t getAction() override;

        // no update needed for this simple policy
        void update(size_t arm, int reward) override { };

        std::string name() const override { return "Uniform"; };

    private:

        mutable std::default_random_engine m_generator;
        size_t m_arms;
};


/* -------------------------------------------------------------------------- */


#endif // __UNIFORM_HPP__
