// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "kl_ucb.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include "common.hpp"


/* -------------------------------------------------------------------------- */


KLUCBStrategy::KLUCBStrategy(unsigned int seed, size_t n_arms) :
    m_generator(seed),
    m_arms(n_arms),
    m_arm_successes(n_arms, 0.0),
    m_arm_visits(n_arms, 0.0),
    m_visits(0.0)
{
}


size_t KLUCBStrategy::getAction() {
    auto unvisited = unvisitedArms();

    // if we have any unvisited arms, pick one uniformly at random
    if (!unvisited.empty()) {
        std::uniform_int_distribution<size_t> randidx(0, unvisited.size()-1);
        return unvisited[randidx(m_generator)];
    }

    // ...otherwise pick the arm with the maximising KL-UCB score
    double best = -std::numeric_limits<double>::infinity();
    size_t best_idx = 0;

    for (size_t i = 0; i < m_arms; i++) {
        double score = klUCB(i);
        if (score > best) {
            best = score;
            best_idx = i;
        }
    }

    return best_idx;
}


void KLUCBStrategy::update(size_t arm, int reward) {
    m_arm_successes[arm] += reward;
    m_arm_visits[arm] += 1.0;
    m_visits += 1.0;
}


void KLUCBStrategy::reset() {
    m_visits = 0.0;
    m_arm_successes = std::vector<double>(m_arms, 0.0);
    m_arm_visits = std::vector<double>(m_arms, 0.0);
}


std::vector<size_t> KLUCBStrategy::unvisitedArms() const {
    std::vector<size_t> rval;

    for (size_t arm=0; arm < m_arms; arm++) {
        if (m_arm_visits[arm] == 0.0)
            rval.emplace_back(arm);
    }

    return rval;
}


double KLUCBStrategy::klUCB(size_t arm) const {
    assert(m_arm_visits[arm] >= 1.0);

    double t = m_visits + 1.0;

    // implementation taken from Bandit Algorithms,
    // Lattimore et al. This is slightly different
    // to the original KL-UCB: https://arxiv.org/abs/1102.2490
    // which has a tunable c parameter.
    auto f = [](double x) {
        double lx = std::log(x);
        return 1.0 + x * lx * lx;
    };

    double ub = std::log(f(t)) / m_arm_visits[arm];
    double p = m_arm_successes[arm] / m_arm_visits[arm];
    double score = maxRelEntropy(p, ub);

    return score;
}


double KLUCBStrategy::maxRelEntropy(double p, double ub) const {
    assert(ub > 0.0);

    // desired precision
    constexpr double eps = 1.0e-8;

    // our initial guess always satisfies the constraint
    // since d(p,p) = 0
    double low = p, high = 1.0;

    // narrow the range using a binary search to desired precision
    do {
        double q = low + (high-low) / 2.0;
        double e = bernoulliRelEntropy(p, q);
        if (e > ub) {
            high = q;
        } else {
            low = q;
        }
    } while ((high - low) > eps);

    return low;
}


/* -------------------------------------------------------------------------- */
