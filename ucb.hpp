#ifndef __UCB_HPP__
#define __UCB_HPP__

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "bandits.hpp"


/* -------------------------------------------------------------------------- */


class UCBStrategy : public BanditStrategy {

    public:

        UCBStrategy(unsigned int seed, size_t n_arms);

        // implement a UCB policy
        size_t getAction() override;

        // update the internal algorithm statistics
        void update(size_t arm, int reward) override;

        std::string name() const override { return "UCB"; }

        // resets the mean/visit statistics
        void reset();

    private:

        // gives a vector of unvisited arms
        std::vector<size_t> unvisitedArms() const;

        // UCB score of a given arm
        double ucb(size_t arm) const;

        // limited amount of randomness used in this implementation
        // so that the "play each arm once" step is done according
        // to a random permutation of the arm indices
        mutable std::default_random_engine m_generator;

        size_t m_arms;
        std::vector<double> m_arm_cumm_reward;
        std::vector<double> m_arm_visits;
        double m_visits;
};


/* -------------------------------------------------------------------------- */


#endif // __UCB_HPP__

