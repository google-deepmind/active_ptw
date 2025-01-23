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

#include "bandits.hpp"

#include <random>
#include <vector>
#include <cstddef>
#include <memory>
#include <utility>
#include <cassert>
#include <iterator>
#include <algorithm>
#include <ostream>

#include "common.hpp"


/* -------------------------------------------------------------------------- */


GeometricAbruptChangeSchedule::GeometricAbruptChangeSchedule(
    double p,
    size_t max_trials,
    unsigned int seed
) :
    m_generator(seed)
{
    std::geometric_distribution<size_t> gdist(p);

    size_t upto = 0;
    do {
        size_t inc = gdist(m_generator);
        upto += inc;

        if (upto < max_trials)
            m_cpts.insert(upto);
    } while (upto < max_trials);
}


bool GeometricAbruptChangeSchedule::changepoint(size_t t) const {
    return m_cpts.find(t) != m_cpts.cend();
}


/* -------------------------------------------------------------------------- */


VectorAbruptChangeSchedule::VectorAbruptChangeSchedule(
    const std::vector<size_t> &times
) :
    m_cpts(times.begin(), times.end())
{}


bool VectorAbruptChangeSchedule::changepoint(size_t t) const {
    return m_cpts.find(t) != m_cpts.cend();
}


/* -------------------------------------------------------------------------- */


TwoPhaseChangeSchedule::TwoPhaseChangeSchedule(
    size_t max_trials,
    arm_initialisation_t thetas_seg1,
    arm_initialisation_t thetas_seg2
) :
    m_halfway(max_trials / 2),
    m_thetas_seg1(thetas_seg1),
    m_thetas_seg2(thetas_seg2)
{}


bool TwoPhaseChangeSchedule::changepoint(size_t t) const {
    return t == m_halfway || t == 1;
}


arm_initialisation_t TwoPhaseChangeSchedule::customArmInitialisation(
  size_t t
) const {
    if (t < m_halfway) {
        return m_thetas_seg1;
    } else {
        return m_thetas_seg2;
    }
}


/* -------------------------------------------------------------------------- */


StochasticBanditProblem::StochasticBanditProblem(
    size_t n_arms,
    unsigned int seed,
    std::unique_ptr<ChangeSchedule> cs
) :
    m_generator(seed),
    m_change_schedule(std::move(cs)),
    m_thetas(n_arms, 0.0)
{
    reset();
}


double StochasticBanditProblem::pull(size_t arm_index) {
  if (arm_index >= m_thetas.size()) {
    die_with_error("invalid arm index");
  }

  m_num_trials++;

  std::bernoulli_distribution coin(m_thetas[arm_index]);
  bool flip = coin(m_generator);
  double r = flip ? 1.0 : 0.0;

  m_cumm_reward += r;

  auto ba = bestArm();
  m_exp_cumm_reward += m_thetas[ba];

  if (m_change_schedule->changepoint(m_num_trials)) {
    auto new_thetas = m_change_schedule->customArmInitialisation(m_num_trials);
    if (new_thetas.empty()) {
      // default to generate thetas uniformly at random
      reset();
    } else {
      assert(new_thetas.size() == m_thetas.size());
      m_thetas = new_thetas;
    }
  }

  return r;
}


void StochasticBanditProblem::reset() {
    std::uniform_real_distribution<double> unit_interval_dist(0.0, 1.0);

    for (size_t i = 0; i < m_thetas.size(); i++) {
        double r = unit_interval_dist(m_generator);
        m_thetas[i] = r;
    }
}


size_t StochasticBanditProblem::bestArm() const {
    return std::distance(
        m_thetas.begin(),
        std::max_element(m_thetas.begin(), m_thetas.end())
    );
}


double StochasticBanditProblem::bestHindsightExpectedReturn() const {
    return m_exp_cumm_reward;
}



size_t StochasticBanditProblem::trials() const {
    return m_num_trials;
}


size_t StochasticBanditProblem::arms() const {
    return m_thetas.size();
}


double StochasticBanditProblem::cummulativeReward() const {
    return m_cumm_reward;
}


bool StochasticBanditProblem::changepoint() const {
    return m_change_schedule->changepoint(trials());
}


std::ostream& operator<<(std::ostream& o, const StochasticBanditProblem& bp) {
    o << "Biases:";
    for (const auto& e : bp.m_thetas) {
        o << " " << e;
    }
    o << std::endl;

    o << "Best arm index: " << bp.bestArm() << std::endl;

    return o;
}


/* -------------------------------------------------------------------------- */


