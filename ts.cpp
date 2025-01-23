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

#include "ts.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "ptw.hpp"


/* -------------------------------------------------------------------------- */


// generate a sample according to a beta distribution
double genBetaSample(
  std::default_random_engine &generator,
  double alpha,
  double beta
) {
  // if X∼Gamma(a, 1), Y∼Gamma(b, 1) then Z = X/(X+Y) ~ Beta(a,b)
  std::gamma_distribution<double> x_gamma_dist(alpha, 1.0);
  std::gamma_distribution<double> y_gamma_dist(beta, 1.0);

  double x, y, z;
  do {
    x = x_gamma_dist(generator);
    y = y_gamma_dist(generator);
    z = x / (x + y);
  } while (z != z);  // FP trick for avoiding infinities

  return z;
}


/* -------------------------------------------------------------------------- */


ThompsonSamplingStrategy::ThompsonSamplingStrategy(
  unsigned int seed, size_t n_arms
) :
  m_generator(seed),
  m_model(n_arms)
{
}


size_t ThompsonSamplingStrategy::getAction() {
  double best = -std::numeric_limits<double>::infinity();
  size_t best_idx = 0;

  for (size_t i = 0; i < m_model.size(); i++) {
    auto ss = m_model[i].posterior();
    double r = genBetaSample(m_generator, ss.first, ss.second);
    if (r > best) {
      best = r;
      best_idx = i;
    }
  }

  return best_idx;
}


void ThompsonSamplingStrategy::update(size_t arm, int reward) {
  m_model[arm].update(reward);
}


/* -------------------------------------------------------------------------- */


ActivePTWBanditStrategy::ActivePTWBanditStrategy(
  unsigned int seed, size_t n_arms
) :
    m_generator(seed),
    m_model(30, n_arms),
    m_arms(n_arms)
{
}


/* sample first a temporal segment according to its posterior weight, then from
   each arms posterior probability, take the argmax as the selected action. */
size_t ActivePTWBanditStrategy::getAction() {
  double best = -std::numeric_limits<double>::infinity();
  size_t best_idx = 0;

  size_t level = levelPosteriorSample();

  for (size_t i = 0; i < m_arms; i++) {
    auto x = m_model.posterior(level, i);
    double r = genBetaSample(m_generator, x.first, x.second);
    if (r > best) {
      best = r;
      best_idx = i;
    }
  }

  return best_idx;
}


void ActivePTWBanditStrategy::update(size_t arm, int reward) {
    m_model.update(reward, arm);
}


size_t ActivePTWBanditStrategy::levelPosteriorSample() const {
  auto lp = m_model.levelPosterior();
  std::discrete_distribution<size_t> level_dist(lp.begin(), lp.end());

  return level_dist(m_generator);
}


std::vector<double> ActivePTWBanditStrategy::levelPosterior() const {
    return m_model.levelPosterior();
}


const ActivePTW &ActivePTWBanditStrategy::model() const {
    return m_model;
}


/* -------------------------------------------------------------------------- */


ParanoidPTWBanditStrategy::ParanoidPTWBanditStrategy(
  unsigned int seed, size_t n_arms
) :
    m_generator(seed),
    m_arms(n_arms),
    m_aptw(seed, n_arms),
    m_trials(0)
{
}


size_t ParanoidPTWBanditStrategy::getAction() {
    constexpr bool UseUniformExploration = true;

    size_t level = m_aptw.levelPosteriorSample();

    // after sampling from the posterior over levels,
    // we see whether we need to do forced exploration,
    // and pick the right rate according to the sampled segment size
    std::uniform_real_distribution<double> uniform01_dist(0.0, 1.0);
    auto lp = m_aptw.levelPosterior();
    size_t k = (lp.size()-1)-level;  // segment size = 2^k
    double clip = std::log(m_trials+1)+1.0;
    while (static_cast<double>(k) > clip) {
        k--;
    }

    if (uniform01_dist(m_generator) < exploreProb(k)) {
        if (UseUniformExploration) {
            std::uniform_int_distribution<size_t> randidx(0, m_arms-1);
            return randidx(m_generator);
        } else {
            return leastExploredArm(level);
        }
    }

  double best = -std::numeric_limits<double>::infinity();
  size_t best_idx = 0;

  for (size_t i = 0; i < m_arms; i++) {
    auto x = m_aptw.model().posterior(level, i);
    double r = genBetaSample(m_generator, x.first, x.second);
    if (r > best) {
      best = r;
      best_idx = i;
    }
  }

  return best_idx;
}


void ParanoidPTWBanditStrategy::update(size_t arm, int reward) {
    m_aptw.update(arm, reward);
    m_trials++;
}


double ParanoidPTWBanditStrategy::exploreProb(size_t log2_segment_size) const {
    constexpr double C = 1.0;

    double k = log2_segment_size;

    double prob = std::pow(2.0, -k) * (std::pow(2.0, k/2) - k * std::log(2.0));

    prob = C * prob;
    prob = std::min(1.0, prob);

    assert(prob >= 0.0 && prob <= 1.0);

    return prob;
}


size_t ParanoidPTWBanditStrategy::leastExploredArm(size_t level) const {
    double best_cnt = std::numeric_limits<double>::infinity();
    size_t best_idx = 0;

    for (size_t arm=0; arm < m_arms; arm++) {
        auto ss = m_aptw.model().posterior(level, arm);
        double cnt = ss.first + ss.second;
        if (cnt < best_cnt) {
            best_cnt = cnt;
            best_idx = arm;
        }
    }

    return best_idx;
}


/* -------------------------------------------------------------------------- */
