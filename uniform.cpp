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

#include "uniform.hpp"

#include <cassert>
#include <cstddef>
#include <random>


/* -------------------------------------------------------------------------- */


UniformSamplingStrategy::UniformSamplingStrategy(
    unsigned int seed,
    size_t n_arms
) :
    m_generator(seed),
    m_arms(n_arms)
{
    assert(m_arms > 0);
}


size_t UniformSamplingStrategy::getAction() {
    std::uniform_int_distribution<size_t> randidx(0, m_arms-1);

    return randidx(m_generator);
}


/* -------------------------------------------------------------------------- */

