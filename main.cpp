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

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"

// bandit algorithms
#include "bandits.hpp"
#include "ts.hpp"
#include "ucb.hpp"
#include "kl_ucb.hpp"
#include "sliding_ucb.hpp"
#include "master.hpp"
#include "constant.hpp"
#include "uniform.hpp"


/* -------------------------------------------------------------------------- */


// container for configuration options (with defaults)
struct params_t {
    unsigned int EnvSeed     = 666;
    unsigned int AgentSeed   = 33;
    size_t       Trials      = 2500;
    size_t       Arms        = 10;
    std::string  Agent       = "ActivePTW";
    std::string  Mode        = "text";
    size_t       PlotRepeats = 400;
    double       CptRate     = 0.002;
    size_t       SWUCBWindow = CptRate > 0 ?
        size_t(1.0 / CptRate + 0.5) : std::numeric_limits<size_t>::max();
    std::string  CptSchedule = "Geometric";
};

// program options
params_t Params;


// process the command line options
void processCmdLine(int argc, char *argv[]) {
    for (size_t i=1; i < argc; i++) {
        std::string s = argv[i];
        auto pos = std::find(s.begin(), s.end(), '=');
        if (pos == s.end()) {
            die_with_error("args need to be in key=value format.");
        }
        std::string lhs(s.begin(), pos);
        std::string rhs(pos+1, s.end());

        if (lhs == "EnvSeed") {
            Params.EnvSeed = std::stoi(rhs);
        } else if (lhs == "AgentSeed") {
            Params.AgentSeed = std::stoi(rhs);
        } else if (lhs == "Trials") {
            Params.Trials = std::stoi(rhs);
            if (Params.Trials < 1) {
                die_with_error("Trials need to be non-zero.");
            }
        } else if (lhs == "PlotRepeats") {
            Params.PlotRepeats = std::stoi(rhs);
            if (Params.PlotRepeats < 1) {
                die_with_error("PlotRepeats need to be positive.");
            }
        } else if (lhs == "SWUCBWindow") {
            Params.SWUCBWindow = std::stoi(rhs);
            if (Params.SWUCBWindow < 1) {
                die_with_error("SWUCBWindow need to be positive.");
            }
        } else if (lhs == "Arms") {
            Params.Arms = std::stoi(rhs);
            if (Params.Arms < 2) {
                die_with_error("Arms needs to be at least 2.");
            }
        } else if (lhs == "Agent") {
            Params.Agent = rhs;
        } else if (lhs == "CptSchedule") {
            Params.CptSchedule = rhs;
        } else if (lhs == "Mode") {
            Params.Mode = rhs;
            if (rhs != "text" && rhs != "plot") {
                die_with_error("Mode needs to be one of text/gui/plot.");
            }
        } else if (lhs == "CptRate") {
            Params.CptRate = std::stod(rhs);
            if (Params.CptRate >= 1.0) {
                die_with_error("CptRate needs to be less than 1.0.");
            }
        } else {
            die_with_error("unrecognised arg.");
        }
    }
}


/* -------------------------------------------------------------------------  */


/* writes a summary of the current state of the bandit problem to stdout */
static void showSummary(const StochasticBanditProblem &bp) {
    double trials = static_cast<double>(bp.trials());
    double regret = bp.bestHindsightExpectedReturn() - bp.cummulativeReward();
    double avg_regret = regret / trials;

    std::cout << bp.trials() << " trials completed." << std::endl;
    std::cout << "Total Reward: " << bp.cummulativeReward() << std::endl;
    std::cout << "Regret: " << regret << std::endl;
    std::cout << "Avg Regret: " << avg_regret << std::endl;
}


/* -------------------------------------------------------------------------- */


// initialise a bandit algorithm from a string
static std::unique_ptr<BanditStrategy> createBanditAlgorithm() {
    auto name = Params.Agent;
    auto seed = Params.AgentSeed;
    auto arms = Params.Arms;
    auto window = Params.SWUCBWindow;

    if (name == "UCB") {
        return std::make_unique<UCBStrategy>(seed, arms);
    } else if (name == "KLUCB") {
        return std::make_unique<KLUCBStrategy>(seed, arms);
    } else if (name == "SWUCB") {
        return std::make_unique<SlidingUCBStrategy>(seed, arms, window);
    } else if (name == "ActivePTW") {
        return std::make_unique<ActivePTWBanditStrategy>(seed, arms);
    } else if (name == "ParanoidPTW") {
        return std::make_unique<ParanoidPTWBanditStrategy>(seed, arms);
    } else if (name == "MALG") {
        return std::make_unique<MalgUCB>(seed, arms, 20);
    } else if (name == "TS") {
        return std::make_unique<ThompsonSamplingStrategy>(seed, arms);
    } else if (name == "Constant") {
        return std::make_unique<ConstantStrategy>(0);
    } else if (name == "Uniform") {
        return std::make_unique<UniformSamplingStrategy>(seed, arms);
    }

    die_with_error("Invalid agent.");

    return nullptr;
}


/* -------------------------------------------------------------------------- */


// create the bandit problem with associated latent change-point schedule
static std::unique_ptr<StochasticBanditProblem> createBanditProblem() {
    if (Params.CptSchedule == "Nasty") {
        arm_initialisation_t theta1(Params.Arms, 0.1);
        theta1[0] = 0.2;
        arm_initialisation_t theta2(Params.Arms, 0.2);
        theta2[0] = 0.2;
        theta2[1] = 0.8;

        return std::make_unique<StochasticBanditProblem>(
            Params.Arms,
            Params.EnvSeed,
            std::make_unique<TwoPhaseChangeSchedule>(
                Params.Trials,
                theta1,
                theta2
            )
        );

    } else if (Params.CptSchedule == "Geometric") {
        return std::make_unique<StochasticBanditProblem>(
            Params.Arms,
            Params.EnvSeed,
            std::make_unique<GeometricAbruptChangeSchedule>(
                Params.CptRate,
                Params.Trials,
                Params.EnvSeed+10007
            )
        );
    }

    die_with_error("Invalid changepoint schedule.");
    return nullptr;
}


/* -------------------------------------------------------------------------  */


// generates some python code which can be run to generate plots
// suitable for academic papers and the like
static int plotMode() {
    std::vector<std::string> agents = {
        "UCB",
        "ActivePTW",
        "MALG",
        "TS",
        "KLUCB",
        "SWUCB",
        "ParanoidPTW"
    };

    std::vector<std::vector<std::vector<double>>> regrets;
    std::vector<size_t> cpts;
    auto original_agent_seed = Params.AgentSeed;

    for (size_t i=0; i < agents.size(); i++) {
        regrets.push_back(std::vector<std::vector<double>>());

        for (size_t j=0; j < Params.PlotRepeats; j++) {
            cpts.clear();

            regrets.back().push_back(std::vector<double>());

            // create the bandit environment
            auto bp = createBanditProblem();

            // create the bandit
            Params.Agent = agents[i];
            Params.AgentSeed = original_agent_seed + j;
            auto agent = createBanditAlgorithm();

            // agent <-> environment loop
            for (size_t t = 0; t < Params.Trials; t++) {
                if (bp->changepoint())
                    cpts.push_back(t+1);

                size_t arm = agent->getAction();
                auto r = bp->pull(arm);
                agent->update(arm, r);

                double regret = bp->bestHindsightExpectedReturn();
                regret -= bp->cummulativeReward();
                regrets[i][j].push_back(regret);
            }
        }
    }

    // write a newline to std::cerr for the progress bar
    std::cerr << std::endl;

    // now generate python code for the plot
    std::cout << "import matplotlib.pyplot as plt" << std::endl;
    std::cout << "import numpy as np" << std::endl;

    // make font size larger
    std::cout << "plt.rcParams.update({'font.size': 50})" << std::endl;

    // write x-axis
    std::cout << "x=np.arange(1," << (Params.Trials+1) << ")" << std::endl;

    // compute mean
    std::vector<std::vector<double>> means;
    for (size_t i=0; i < agents.size(); i++) {
        means.push_back(std::vector<double>());

        for (size_t t=1; t <= Params.Trials; t++) {
            double total = 0.0;
            for (size_t j=0; j < Params.PlotRepeats; j++) {
                total += regrets[i][j][t-1];
            }
            means.back().push_back(total / Params.PlotRepeats);
        }
    }

    // compute CI
    std::vector<std::vector<double>> cis;
    for (size_t i=0; i < agents.size(); i++) {
        cis.push_back(std::vector<double>());

        for (size_t t=1; t <= Params.Trials; t++) {
            double total = 0.0;
            for (size_t j=0; j < Params.PlotRepeats; j++) {
                total += std::pow(regrets[i][j][t-1] - means[i][j], 2.0);;
            }
            double repeats = static_cast<double>(Params.PlotRepeats);
            double var = total / (repeats-1.0);
            double stddev   = std::sqrt(var);
            double stderr = stddev / std::sqrt(repeats);
            cis.back().push_back(1.96*stderr);
        }
    }

    // write datapoints
    for (size_t i=0; i < agents.size(); i++) {
        std::cout << "y" << i << "= np.asarray([";
        for (size_t t=1; t <= Params.Trials; t++) {
            std::cout << means[i][t-1] << ", " << std::endl;
        }
        std::cout << "])" << std::endl;

        std::cout << "y" << i << "u= np.asarray([";
        for (size_t t=1; t <= Params.Trials; t++) {
            std::cout << (means[i][t-1]+cis[i][t-1]) << ", " << std::endl;
        }
        std::cout << "])" << std::endl;

        std::cout << "y" << i << "b= np.asarray([";
        for (size_t t=1; t <= Params.Trials; t++) {
            std::cout << (means[i][t-1]-cis[i][t-1]) << ", " << std::endl;
        }
        std::cout << "])" << std::endl;
    }

    for (size_t i=0; i < agents.size(); i++) {
        std::cout << "plt.plot(x, y";
        std::cout << i;
        std::cout << ", label='" << agents[i] << "')" << std::endl;

        std::cout << "plt.fill_between(x, y";
        std::cout << i << "b, y" << i << "u,";
        std::cout << " alpha=.15)" << std::endl;
    }

    // write labels
    std::cout << "plt.plot()" << std::endl;
    std::cout << "plt.xlabel('Time')" << std::endl;
    std::cout << "plt.ylabel('Regret')" << std::endl;
    std::cout << "plt.title('Regret vs Time ";
    std::cout << "[Actions=" << Params.Arms;
    if (Params.CptSchedule != "Nasty")
        std::cout << ", " << "CptRate=" << Params.CptRate;
    std::cout << "]";
    std::cout << "')" << std::endl;
    std::cout << "plt.legend()" << std::endl;

    // write changepoints
    for (auto &e : cpts) {
        std::cout << "plt.axvline(x=" << e;
        std::cout << ", dashes=[0.1,0.5])" << std::endl;
    }

    std::cout << "plt.show()" << std::endl;

    return 0;
}


/* -------------------------------------------------------------------------- */


/* run the algorithm in text mode. */
static int textMode() {
    // create the bandit environment
    auto bp = createBanditProblem();

    // create the bandit
    auto agent = createBanditAlgorithm();

    // agent <-> environment loop
    for (size_t t = 0; t < Params.Trials; t++) {
        size_t arm = agent->getAction();
        auto r = bp->pull(arm);
        agent->update(arm, r);
    }

    showSummary(*bp);

    return 0;
}


/* -------------------------------------------------------------------------  */


/* application entry point */
int main(int argc, char* argv[]) {
    processCmdLine(argc, argv);

    if (Params.Mode == "text") {
        return textMode();
    } else if (Params.Mode == "plot") {
        return plotMode();
    }

    return 0;
}


/* -------------------------------------------------------------------------- */

