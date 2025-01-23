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

#include "ucb.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>


/* -------------------------------------------------------------------------- */


UCBStrategy::UCBStrategy(unsigned int seed, size_t n_arms) :
  m_generator(seed),
  m_arms(n_arms),
  m_arm_cumm_reward(n_arms, 0.0),
  m_arm_visits(n_arms, 0.0),
  m_visits(0.0)
{
}


void UCBStrategy::reset() {
  m_visits = 0.0;
  m_arm_cumm_reward = std::vector<double>(m_arms, 0.0);
  m_arm_visits = std::vector<double>(m_arms, 0.0);
}


size_t UCBStrategy::getAction() {
  auto unvisited = unvisitedArms();

  // if we have any unvisited arms, pick one uniformly at random
  if (!unvisited.empty()) {
    std::uniform_int_distribution<size_t> randidx(0, unvisited.size()-1);
    return unvisited[randidx(m_generator)];
  }

  // ...otherwise pick the arm with the maximising UCB score
  double best = -std::numeric_limits<double>::infinity();
  size_t best_idx = 0;

  for (size_t i = 0; i < m_arms; i++) {
    double score = ucb(i);
    if (score > best) {
      best = score;
      best_idx = i;
    }
  }

  return best_idx;
}


void UCBStrategy::update(size_t arm, int reward) {
    m_arm_cumm_reward[arm] += reward;
    m_arm_visits[arm] += 1.0;
    m_visits += 1.0;
}


std::vector<size_t> UCBStrategy::unvisitedArms() const {
  std::vector<size_t> rval;

  for (size_t arm=0; arm < m_arms; arm++) {
    if (m_arm_visits[arm] == 0.0)
      rval.emplace_back(arm);
  }

  return rval;
}


double UCBStrategy::ucb(size_t arm) const {
  double mean = m_arm_cumm_reward[arm] / m_arm_visits[arm];
  double ci   = std::sqrt((2.0 * std::log(m_visits)) / m_arm_visits[arm]);

  return mean + ci;
}


/* -------------------------------------------------------------------------- */


