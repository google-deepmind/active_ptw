#ifndef __CONSTANT_HPP__
#define __CONSTANT_HPP__


#include "bandits.hpp"

#include <string>
#include <cstddef>


/* -------------------------------------------------------------------------- */


// an interface describing a constant action strategy
class ConstantStrategy : public BanditStrategy {

    public:

        ConstantStrategy(size_t action) : m_action(action) {}

        // get the action from the bandit algorithm
        size_t getAction() override { return m_action; }

        // update the algorithms internal state after
        // pulling an arm and receiving a reward
        void update(size_t arm, int reward) override { }

        // name of the method, e.g. UCB
        std::string name() const override { return "Constant"; }

    private:

        size_t m_action;
};


/* -------------------------------------------------------------------------- */


#endif // __CONSTANT_HPP__

