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

#include "master.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>


/* -------------------------------------------------------------------------- */


MalgUCB::MalgUCB(
    unsigned int seed,
    size_t n_arms,
    size_t depth
) :
    m_generator(seed),
    m_seed(seed),
    m_arms(n_arms),
    m_n(depth),
    m_tau(1)
{
    for (size_t i=0; i < m_n+1; i++) {
        m_instances.emplace_back(std::unique_ptr<instance_t>(nullptr));
    }
}


size_t MalgUCB::getAction() {
    // handle the reseting schedule
    for (size_t off=0; off <= m_n; off++) {
        size_t m = m_n-off;

        // if m_tau is a multiple of 2^m
        if (((m_tau-1) % (1 << m)) == 0) {
            double threshold = rho(std::pow(2.0, m_n)) / rho(std::pow(2.0, m));

            std::uniform_real_distribution<double> uniform01_dist(0.0, 1.0);
            if (uniform01_dist(m_generator) < threshold) {
                size_t start = m_tau;
                size_t end   = m_tau + (1 << m) - 1;

                // reset the UCB instance
                if (m_instances[m].get() == nullptr) {
                    m_instances[m] = std::make_unique<instance_t>(
                        m_seed + m,  // use different seeds for different levels
                        m_arms, start, end
                    );
                } else {
                    m_instances[m]->s = start;
                    m_instances[m]->e = end;
                    m_instances[m]->alg.reset();
                }
            }
        }
    }

    size_t active_idx = activeInstance();

    return m_instances[active_idx]->alg.getAction();
}


void MalgUCB::update(size_t arm, int reward) {
    size_t active_idx = activeInstance();
    m_instances[active_idx]->alg.update(arm, reward);
    m_tau++;
}


double MalgUCB::rho(double t) const {
    double a = static_cast<double>(m_arms);
    return std::sqrt(a/t) + a/t;
}


size_t MalgUCB::activeInstance() const {
    // find the smallest active segment
    size_t best = std::numeric_limits<size_t>::max();
    size_t best_idx = -1;

    for (size_t i=0; i < m_instances.size(); i++) {
        auto &e = m_instances[i];

        if (e.get() != nullptr && m_tau >= e->s && m_tau <= e->e) {
            if (e->length() < best) {
                best = e->length();
                best_idx = i;
            }
        }
    }

    // best index should always exist by definition
    assert(best != std::numeric_limits<size_t>::max());

    return best_idx;
}


/* -------------------------------------------------------------------------- */


size_t MasterUCB::getAction() {
    // todo
    return 0;
}


void MasterUCB::update(size_t arm, int reward) {
}


/* -------------------------------------------------------------------------- */

