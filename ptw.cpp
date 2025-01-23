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

#include "ptw.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <vector>

#include "common.hpp"


/* -------------------------------------------------------------------------- */


/* PTW constructor */
ActivePTW::ActivePTW(size_t depth, size_t arms) :
    m_index(0),
    m_nodes(depth + 1, ActivePTWNode_t(arms)),
    m_depth(depth),
    m_arms(arms)
{
    double a = static_cast<double>(arms);
    double x = (a-1.0)/a;
    LogStopWeight = std::log(x);
    LogSplitWeight = std::log(1.0 - x);
}


/* the probability of seeing a particular symbol next */
double ActivePTW::prob(int r, size_t k) {
    auto post = levelPosterior();

    std::vector<double> probs;
    probs.reserve(post.size());
    for (size_t i = 0; i < post.size(); i++) {
        probs.emplace_back(m_nodes[i].prob(r, k));
    }

    return std::inner_product(post.begin(), post.end(), probs.begin(), 0.0);
}


/* process a new piece of sensory experience */
void ActivePTW::update(int r, size_t k) {
    assert(m_index < (1 << m_depth));

    // mscb requires the current 1-based time
    size_t i = mscb(m_index + 1);

    // save weighted probability in change point's parent
    m_nodes[i].m_log_buf = m_nodes[i + 1].m_log_weighted;

    // now reset statistics from the change point downwards
    for (size_t j = i + 1; j <= m_depth; j++) {
        m_nodes[j] = ActivePTWNode_t(m_arms);
    }

    // compute weighted probability from bottom up
    ActivePTWNode_t& n = m_nodes[m_depth];
    n.m_model[k].update(r);
    n.m_log_weighted = n.logMarginal();

    for (size_t i = 1; i <= m_depth; i++) {
        size_t idx = m_depth - i;
        m_nodes[idx].m_model[k].update(r);
        double lhs = LogStopWeight + m_nodes[idx].logMarginal();
        double rhs = LogSplitWeight;
        rhs += m_nodes[idx + 1].m_log_weighted;
        rhs += m_nodes[idx].m_log_buf;
        m_nodes[idx].m_log_weighted = logAdd(lhs, rhs);
    }

    m_index++;
}


/* the number of bits to the left of the most significant
   location at which times t-1 and t-2 differ, where t is
   the 1 based current time. */
size_t ActivePTW::mscb(index_t t) const {
    if (t == 1) return 0;

    size_t c = m_depth - 1;
    size_t cnt = 0;

    for (index_t i = 0; i < m_depth; i++) {
        index_t tm1 = t - 1, tm2 = t - 2;
        index_t mask = static_cast<uint64_t>(1) << c;

        if ((tm1 & mask) != (tm2 & mask)) return cnt;

        c--, cnt++;
    }

    return cnt;
}


/* Compute the posterior weights for current temporal discretization level. */
std::vector<double> ActivePTW::levelPosterior() const {
    double posterior_mass_left = 1.0;

    std::vector<double> dest;

    // compute the posterior weights of each level from top down
    for (size_t i = 0; i <= m_depth; i++) {
        // compute log posterior of stopping at level i
        double x = LogStopWeight + m_nodes[i].logMarginal();
        x -= m_nodes[i].m_log_weighted;
        double stop_post = std::exp(x);

        dest.push_back(posterior_mass_left * stop_post);
        posterior_mass_left *= (1.0 - stop_post);

        assert(dest.back() >= 0.0 && dest.back() <= 1.0);

        // for numerical stability
        posterior_mass_left = std::max(posterior_mass_left, 0.0);
        assert(posterior_mass_left >= 0.0 && posterior_mass_left <= 1.0);
    }

    assert(dest.size() == m_depth + 1);

    return dest;
}


beta_suff_stats_t ActivePTW::posterior(size_t level, size_t arm_index) const {
    return m_nodes[level].m_model[arm_index].posterior();
}


/* -------------------------------------------------------------------------- */

