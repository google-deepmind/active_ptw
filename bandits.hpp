#ifndef __BANDITS_HPP__
#define __BANDITS_HPP__

#include <cstddef>
#include <limits>
#include <memory>
#include <ostream>
#include <random>
#include <set>
#include <string>
#include <vector>


/* -------------------------------------------------------------------------- */


// an interface describing a bandit strategy
class BanditStrategy {

    public:

        // get the action from the bandit algorithm
        virtual size_t getAction() = 0;

        // update the algorithms internal state after
        // pulling an arm and reeiving a reward
        virtual void update(size_t arm, int reward) = 0;

        // name of the method, e.g. UCB
        virtual std::string name() const = 0;

        virtual ~BanditStrategy() = default;
};


/* -------------------------------------------------------------------------- */


// bandit tracking statistics
struct bandit_stats_t {
    std::vector<double> cummulative_reward;
    std::vector<double> regret;
    std::vector<size_t> changepts;
    size_t max_trials = std::numeric_limits<size_t>::max();
};


/* -------------------------------------------------------------------------- */


using arm_initialisation_t = std::vector<double>;


// interface for various types of changing sources
class ChangeSchedule {

    public:

        // does the underlying environment change at this point
        virtual bool changepoint(size_t t) const = 0;

        // allows the specification of custom assignments of arm parameters,
        // defaulting to an empty vector which applies no additional change
        // from the default mechanism specified by the bandit problem
        virtual arm_initialisation_t customArmInitialisation(size_t t) const {
            return arm_initialisation_t();
        }

        virtual ~ChangeSchedule() = default;
};


// schedule which gives rise to a stationary problem
class NoChangeSchecule : public ChangeSchedule {

    public:

        bool changepoint(size_t) const override { return false; }

};


// generate a sequence of geometrically spaced change-points
class GeometricAbruptChangeSchedule : public ChangeSchedule {

    public:

        GeometricAbruptChangeSchedule(
            double p,
            size_t max_trials,
            unsigned int seed
        );

        // checks if a changepoint at t with log(t) time complexity
        bool changepoint(size_t) const override;

    private:

        mutable std::default_random_engine m_generator;
        std::set<size_t> m_cpts;
};


// describe a change point schedule by an unordered list of indices
class VectorAbruptChangeSchedule : public ChangeSchedule {

    public:

        VectorAbruptChangeSchedule(const std::vector<size_t> &times);

        // checks if a changepoint at t with log(t) time complexity
        bool changepoint(size_t t) const override;

    private:

        std::set<size_t> m_cpts;
};


// an adversarially chosen change-point scenario which penalises algorithms
// who perform well in the stationary case. construction uses two equal sized
// segments, the best arm in the first segment will maintain its value in the
// second segment, but in the second segment will no longer be optimal.
// construction inspired from Thm 31.2 in Bandit Algorithms by Lattimore et al.
class TwoPhaseChangeSchedule : public ChangeSchedule {

    public:

        TwoPhaseChangeSchedule(
            size_t max_trials,
            arm_initialisation_t thetas_seg1,
            arm_initialisation_t thetas_seg2
        );

        bool changepoint(size_t t) const override;

        arm_initialisation_t customArmInitialisation(size_t t) const override;

    private:

        size_t m_halfway;
        arm_initialisation_t m_thetas_seg1;
        arm_initialisation_t m_thetas_seg2;
};


/* -------------------------------------------------------------------------- */


// a bernoulli stochastic bandit problem, parametrised by a change-point policy
class StochasticBanditProblem {

    friend std::ostream& operator<<(
      std::ostream& o,
      const StochasticBanditProblem& bp
    );

    public:

        // constructs a new bernoulli stochastic bandit problem,
        // defaults to stationary case but can be parametrised with a
        // ChangeSchedule
        StochasticBanditProblem(
            size_t n_arms,
            unsigned int seed,
            std::unique_ptr<ChangeSchedule> cs =
                std::make_unique<NoChangeSchecule>()
        );

        // pull an arm, receive reward
        double pull(size_t arm_index);

        // total number of times any arm is pulled
        size_t trials() const;

        // the number of arms in the bandit problem
        size_t arms() const;

        // how much reward has been accumulated so far by pulling an arm
        double cummulativeReward() const;

        // the beta arm with full knowledge of the latents
        size_t bestArm() const;

        // reset the underlying true reward distribution
        void reset();

        // the expected return of always playing the best arm at each time step,
        // used to calculate regret
        double bestHindsightExpectedReturn() const;

        // did a change just occur at the current timestep
        bool changepoint() const;

    private:

        mutable std::default_random_engine m_generator;

        std::unique_ptr<ChangeSchedule> m_change_schedule;

        size_t m_num_trials = 0;
        double m_cumm_reward = 0.0;

        std::vector<double> m_thetas;

        double m_exp_cumm_reward = 0.0;
};


/* -------------------------------------------------------------------------- */


#endif // __BANDITS_HPP__

