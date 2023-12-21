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


def init_random_configs():
    configs = sorted([f.absolute() for f in DATA_DIR.glob("*.pkl")], key=sort_configs)
    n_existing = len(configs)

    for i in range(N_GENERATIONS_PER_EVOLUTION - n_existing):
        nn = NeuralNetwork.random(n_inputs=2, n_hidden=3, n_outputs=2)
        nn_fname = (DATA_DIR / f"nn_{i + n_existing}.pkl").absolute()
        nn.to_file(nn_fname)
        configs.append(nn_fname)
    configs.reverse()
    return configs


def sort_configs(path):
    return int(
        str(path).replace(str(DATA_DIR), "").replace("/nn_", "").replace(".pkl", "")
    )


def bots_from_config(nn_fname):
    global UNIQUE_ID

    n_gen = str(nn_fname.name).replace(".pkl", "").replace("nn_", "")
    nn_gen = int(n_gen)

    if nn_gen != GENERATION_COUNT:
        raise Exception(f"Wrong generation! {nn_gen} {GENERATION_COUNT}")

    for _ in range(0, N_ROBOTS_PER_GENERATION):
        x = random.randint(-11, 11) / 10
        y = random.randint(-11, 11) / 10
        rot_z = random.randint(-10, 10) / 10

        children_field.importMFNodeFromString(
            -1,
            f"""
            DEF braitenberg_vehicle_{UNIQUE_ID} BraitenbergLightBot {{
                translation {x} {y} 0 
                rotation 0 0 1 {rot_z}
                color 0 1 0 
                controller "swarm_ga_fl"
                name "braitenberg_vehicle_{UNIQUE_ID}"
                customData "{nn_fname}"
            }}
            """,
        )

        UNIQUE_ID += 1


def init_configs_from_evolution(generation_scores_combined):
    top_scores = sorted(
        generation_scores_combined.items(), key=lambda x: x[1], reverse=True
    )

    print(f"Top scores: {top_scores[:3]}")

    top_3_nn = [
        NeuralNetwork.from_file(DATA_DIR / f"nn_{i}.pkl") for i, _ in top_scores[:3]
    ]

    top_3_mutated_strong = [deepcopy(nn).mutate(rate=2, prob=1) for nn in top_3_nn]
    top_3_mutated_modest = [deepcopy(nn).mutate(rate=1.5, prob=0.75) for nn in top_3_nn]
    top_3_mutated_mild = [deepcopy(nn).mutate(rate=1, prob=0.5) for nn in top_3_nn]
    top_3_mutated_weak = [deepcopy(nn).mutate(rate=0.75, prob=0.25) for nn in top_3_nn]
    top_3_mutated_minimal = [deepcopy(nn).mutate(rate=0.5, prob=0.1) for nn in top_3_nn]

    mutated = [
        *top_3_mutated_strong,
        *top_3_mutated_modest,
        *top_3_mutated_mild,
        *top_3_mutated_weak,
        *top_3_mutated_minimal,
    ]

    rest = (
        [
            NeuralNetwork.random(n_inputs=2, n_hidden=3, n_outputs=2)
            for _ in range(N_GENERATIONS_PER_EVOLUTION - len(mutated))
        ]
        if N_GENERATIONS_PER_EVOLUTION > len(mutated)
        else []
    )

    configs = []

    for i, nn in enumerate(
        [
            *rest,
        ]
    ):
        nn_fname = DATA_DIR / f"nn_{GENERATION_COUNT + i + 1}.pkl"
        nn.to_file(nn_fname)
        configs.append(nn_fname)

    configs.reverse()

    return configs


def get_fitness_scores(braits):
    print("Computing fitness scores...")
    return [
        float(b.getField("customData").getSFString())
        if b.getField("customData").getSFString()
        else 0
        for b in braits
    ]


def compute_generation_fitness():
    braits = get_current_bot_nodes()

    gen_scores_avg = np.mean(get_fitness_scores(braits))

    GENERATION_SCORES_COMBINED[GENERATION_COUNT] = gen_scores_avg
    print(f"Generation {GENERATION_COUNT} score: {gen_scores_avg}")


def remove_old_bots():
    for robot in get_current_bot_nodes():
        robot.remove()


def get_current_bot_nodes():
    return [
        supervisor.getFromDef(f"braitenberg_vehicle_{i}")
        for i in range(UNIQUE_ID - N_ROBOTS_PER_GENERATION, UNIQUE_ID)
    ]


def delete_old_configs():
    for config in DATA_DIR.glob("*.pkl"):
        gen_count = int(
            str(config)
            .replace(str(DATA_DIR), "")
            .replace("/nn_", "")
            .replace(".pkl", "")
        )

        top_3_gens = sorted(
            GENERATION_SCORES_COMBINED.items(), key=lambda x: x[1], reverse=True
        )[:3]
        top_3_gens = [gen for gen, _ in top_3_gens]

        if (
            gen_count < GENERATION_COUNT - N_GENERATIONS_PER_EVOLUTION * 2
            and gen_count not in top_3_gens
        ):
            config.unlink()


N_ROBOTS_PER_GENERATION = 15
N_GENERATIONS_PER_EVOLUTION = 20
SECONDS_PER_GENERATION = 180

GENERATION_COUNT = 0
GENERATION_SCORES_COMBINED = {}
UNIQUE_ID = 0

# start with random configs
NN_CONFIGS = init_random_configs()
bots_from_config(NN_CONFIGS.pop())

RESET = False

while supervisor.step(timeStep) != -1:
    if RESET:
        RESET = False
        GENERATION_COUNT += 1
        bots_from_config(NN_CONFIGS.pop())
        supervisor.simulationSetMode(supervisor.SIMULATION_MODE_FAST)

    if supervisor.getTime() > SECONDS_PER_GENERATION:
        RESET = True

        # save scores before creating new generation of bots
        compute_generation_fitness()
        remove_old_bots()

        if len(NN_CONFIGS) == 0:
            NN_CONFIGS = init_configs_from_evolution(GENERATION_SCORES_COMBINED)
            delete_old_configs()

        supervisor.simulationReset()  # will always happen at END of step!
