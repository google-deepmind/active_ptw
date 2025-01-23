#ifndef __SLIDING_UCB_HPP__
#define __SLIDING_UCB_HPP__

#include <cstddef>
#include <deque>
#include <random>
#include <string>
#include <vector>

#include "bandits.hpp"


/* -------------------------------------------------------------------------- */

// an implementation of Sliding Window UCB
//      https://arxiv.org/pdf/0805.3415

class SlidingUCBStrategy : public BanditStrategy {

    public:

        // SlidingWindow-UCB for a given window size
        SlidingUCBStrategy(unsigned int seed, size_t n_arms, size_t window);

        // implement a UCB policy
        size_t getAction() override;

        // update the internal algorithm statistics
        void update(size_t arm, int reward) override;

        std::string name() const override { return "SlidingUCB"; }

        // resets the mean/visit statistics
        void reset();

    private:

        // gives a vector of unvisited arms
        std::vector<size_t> unvisitedArms() const;

        // windowed UCB score of a given arm
        double ucb(size_t arm) const;

        // limited amount of randomness used in this implementation
        // so that the "play each arm once" step is done according
        // to a random permutation of the arm indices
        mutable std::default_random_engine m_generator;

        size_t m_arms;
        size_t m_window;
        std::deque<size_t> m_plays;
        std::deque<double> m_rewards;
        std::vector<double> m_arm_cumm_reward;
        std::vector<double> m_arm_visits;
};


/* -------------------------------------------------------------------------- */


#endif // __SLIDING_UCB_HPP__

