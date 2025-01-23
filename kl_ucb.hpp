#ifndef __KL_UCB_HPP__
#define __KL_UCB_HPP__


#include "bandits.hpp"


/* -------------------------------------------------------------------------- */


class KLUCBStrategy : public BanditStrategy {

    public:

        KLUCBStrategy(unsigned int seed, size_t n_arms);

        // implement the KL-UCB policy
        size_t getAction() override;

        // update the internal algorithm statistics
        void update(size_t arm, int reward) override;

        std::string name() const override { return "KL-UCB"; }

        // resets the mean/visit statistics
        void reset();

    private:

        // gives a vector of unvisited arms
        std::vector<size_t> unvisitedArms() const;

        // KL-UCB score of a given arm
        double klUCB(size_t arm) const;

        // maximise bernoulli relative entropy d(p, q) <= ub w.r.t. q
        double maxRelEntropy(double p, double ub) const;

        // limited amount of randomness used in this implementation
        // so that the "play each arm once" step is done according
        // to a random permutation of the arm indices
        mutable std::default_random_engine m_generator;

        size_t m_arms;
        std::vector<double> m_arm_successes;
        std::vector<double> m_arm_visits;
        double m_visits;
};


/* -------------------------------------------------------------------------- */


#endif // __UCB_HPP__

