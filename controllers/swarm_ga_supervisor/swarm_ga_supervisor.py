import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from controller import Supervisor

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

sys.path.append(str(BASE_DIR / "scripts"))

from neural import Fitness, NeuralNetwork

# Get reference to the robot.
supervisor = Supervisor()

# Get simulation step length.
timeStep = int(supervisor.getBasicTimeStep())

# Set simulation speed fastest
supervisor.simulationSetMode(supervisor.SIMULATION_MODE_FAST)

root_node = supervisor.getRoot()
children_field = root_node.getField("children")
braits = []


def init_random_configs():
    configs = []
    for i in range(N_GENERATIONS_PER_EVOLUTION):
        nn = NeuralNetwork.random()
        nn_fname = DATA_DIR / f"nn_{GENERATION_COUNT+i}.pkl"
        nn.to_file(nn_fname)
        configs.append(nn_fname)
    configs.reverse()
    return configs


def bot_from_config(nn_fname):
    braits.clear()

    n_gen = str(nn_fname.name).replace(".pkl", "").replace("nn_", "")
    nn_gen = int(n_gen)

    if nn_gen != GENERATION_COUNT:
        raise Exception(f"Wrong generation! {nn_gen} {GENERATION_COUNT}")

    for i in range(0, N_ROBOTS_PER_GENERATION):
        x = random.randint(-10, 10) / 10
        y = random.randint(-10, 10) / 10
        rot_z = random.randint(-10, 10) / 10

        children_field.importMFNodeFromString(
            -1,
            f"""
            DEF braitenberg_vehicle_{i} BraitenbergLightBot {{
                translation {x} {y} 0 
                rotation 0 0 1 {rot_z}
                color 0 1 0 
                controller "swarm_ga"
                name "braitenberg_vehicle_{i}"
                customData "{nn_fname}"
            }}
            """,
        )

        braits.append(supervisor.getFromDef(f"braitenberg_vehicle_{i}"))


def init_configs_from_evolution(generation_scores_combined):
    top_scores = sorted(
        generation_scores_combined.items(), key=lambda x: x[1], reverse=True
    )

    print(f"Top scores: {top_scores[:3]}")

    top_3_nn = [
        NeuralNetwork.from_file(DATA_DIR / f"nn_{i}.pkl") for i, _ in top_scores[:3]
    ]

    top_3_mutated = [deepcopy(nn).mutate() for nn in top_3_nn]

    crossover_one = deepcopy(top_3_nn[0]).crossover(top_3_nn[0])
    crossover_two = deepcopy(top_3_nn[0]).crossover(top_3_nn[1])
    crossover_three = deepcopy(top_3_nn[0]).crossover(top_3_nn[2])
    crossover_four = deepcopy(top_3_nn[1]).crossover(top_3_nn[2])

    rest = [NeuralNetwork.random() for _ in range(N_GENERATIONS_PER_EVOLUTION - 10)]

    configs = []

    for i, nn in enumerate(
        [
            *top_3_nn,
            *top_3_mutated,
            crossover_one,
            crossover_two,
            crossover_three,
            crossover_four,
            *rest,
        ]
    ):
        nn_fname = DATA_DIR / f"nn_{GENERATION_COUNT + i}.pkl"
        nn.to_file(nn_fname)
        configs.append(nn_fname)

    configs.reverse()
    return configs


def get_fitness_scores(braits):
    return [
        float(b.getField("customData").getSFString())
        if b.getField("customData").getSFString()
        else 0
        for b in braits
    ]


N_ROBOTS_PER_GENERATION = 35
N_GENERATIONS_PER_EVOLUTION = 15
SECONDS_PER_GENERATION = 90

GENERATION_COUNT = 0
GENERATION_SCORES = {}
GENERATION_SCORES_COMBINED = {}

# start with 10 random configs
NN_CONFIGS = init_random_configs()
INIT = True

while supervisor.step(timeStep) != -1:
    if INIT:
        bot_from_config(NN_CONFIGS.pop())
        GENERATION_COUNT += 1
        INIT = False

    if supervisor.getTime() > SECONDS_PER_GENERATION:
        # save scores before creating new generation of bots
        gen_scores = get_fitness_scores(braits)
        GENERATION_SCORES[GENERATION_COUNT] = gen_scores
        GENERATION_SCORES_COMBINED[GENERATION_COUNT] = np.mean(gen_scores)

        print(f"Generation {GENERATION_COUNT} score: {np.mean(gen_scores)}")

        supervisor.simulationReset()
        INIT = True

    if len(NN_CONFIGS) == 0:
        NN_CONFIGS = init_configs_from_evolution(GENERATION_SCORES_COMBINED)
