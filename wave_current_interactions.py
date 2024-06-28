from module_utils import *
import signal
from module_preparations import create_experiment, load_dump, NumericalProblem
from module_numerical_experiment_zero_singular import getting_statistics, MetaData
from collections.abc import Iterable
import argparse
import importlib
from pathlib import Path

StrPath = str

def handle_sigterm(signum, frame) -> None:
    signame = signal.Signals(signum).name
    print(f"Signal handler called with signal {signame} ({signum})")
    raise KeyboardInterrupt

def _verify_files(filename: StrPath, suffix: str):
    if not filename.endswith(suffix):
        raise TypeError(f"{suffix}-file should be passed")
    path_to_file = Path(filename)
    if not path_to_file.exists():
        raise FileNotFoundError(f"File {filename} not found") 
    if not path_to_file.is_file():
        raise TypeError(f"{filename} is not file")
    

def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog = "wave_current_interactions.py")
    subparsers = parser.add_subparsers(dest = 'subcommand')
    subparsers.required = True  

    parser_sym = subparsers.add_parser("symbolic")

    parser_num = subparsers.add_parser("numerical")
    parser_num._action_groups.pop()
    required = parser_num.add_argument_group("Required arguments")
    required.add_argument('-p', "--pkl", required = True, type = StrPath,
                          help = "Path to .pkl-file with numerical_problem class instance")
    required.add_argument('-c', "--config", required = True, type = StrPath,
                          help = "Path to .conf-file with requared configurations")
    optional = parser_num.add_argument_group("Optional arguments")
    optional.add_argument("--show_progress", required = False, action = "store_true",
                          help = "Whether to show progress bar")
    
    cli_args = parser.parse_args()
    return cli_args


def initialize_numerical_experiment(cli_args) -> tuple[Iterable | float, Iterable| float, NumericalProblem, bool]:
    _, pkl_filename, config_filename, show_progress = vars(cli_args).values()
    _verify_files(pkl_filename, suffix = ".pkl")
    _verify_files(config_filename, suffix = ".py")
    config = config_filename[:-3]
    try:
        config = Path(config).as_posix()
        conf = importlib.import_module(config)
    except:
        raise ModuleNotFoundError("Configuration file is corrupted")
    try:
        alpha_range = conf.alpha_range
    except AttributeError:
        raise AttributeError(f"Configuration file is currupted: no attribute named alpha_range")
    try:
        theta_range = conf.theta_range
    except AttributeError:
        raise AttributeError(f"Configuration file is currupted: no attribute named theta_range")
    try:
        num_mu_range = conf.num_mu_range
    except AttributeError:
        raise AttributeError(f"Configuration file is currupted: no attribute named num_mu_range")
    try:
        numerical_problem = load_dump(pkl_filename)
    except:
        raise TypeError(f".pkl-file is corrupted")
    return alpha_range, theta_range, num_mu_range, numerical_problem, show_progress


def save_storage(storage: list[MetaData], filename: str) -> None:
    with open(filename, 'a') as storage_fp:
        data = ''.join("{:.3e}\t{:.3e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\n".format(*metadata) for metadata in storage)
        storage_fp.write(data)


def perform_numerical_experiment(alpha_range: Iterable| float, 
                                 theta_range: Iterable| float, 
                                 num_mu_range: Iterable, 
                                 numerical_problem: NumericalProblem, 
                                 show_progress: bool,
                                 N_num_mu_points: int = 200):
    gen =  getting_statistics(numerical_problem, theta_range, alpha_range, num_mu_range, N_num_mu_points, show_progress)
    storage = []
    storage_filename = "storage.data"
    count = 0
    try:
        for metadata in gen:
            storage.append(metadata)
            if (count % 16) == 0:
                save_storage(storage, storage_filename)
                storage = []
    except KeyboardInterrupt:
        if storage:
            save_storage(storage, storage_filename)
        raise 


def main():
    cli_args = parse_cli()
    if cli_args.subcommand == "symbolic":
        create_experiment(h0 = 1e-2, epsilon = 0.05)
    if cli_args.subcommand == "numerical":
        alpha_range, theta_range, num_mu_range, numerical_problem, show_progress = initialize_numerical_experiment(cli_args)
        perform_numerical_experiment(alpha_range, theta_range, num_mu_range, numerical_problem, show_progress)


if __name__ == "__main__":
    main()
    
