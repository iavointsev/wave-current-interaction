from module_utils import *
import signal
from module_preparations import create_experiment, load_dump, NumericalProblem, NumericalParameters
from collections.abc import Iterable
import module_numerical_experiment_zero_singular as zs
import module_numerical_experiment_newton as newt
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
    parser_zero_singular = subparsers.add_parser("zero_singular")
    parser_newton = subparsers.add_parser("newton")

    for subparser in (parser_zero_singular, parser_newton):
        subparser._action_groups.pop()
        required = subparser.add_argument_group("Required arguments")
        required.add_argument('-p', "--pkl", required = True, type = StrPath,
                            help = "Path to .pkl-file with numerical_problem class instance")
        required.add_argument('-c', "--config", required = True, type = StrPath,
                            help = "Path to .conf-file with requared configurations")
        optional = subparser.add_argument_group("Optional arguments")
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


def _save_storage(storage: list[zs.MetaData | newt.MetaData], filename: str) -> None:
    with open(filename, 'a') as storage_fp:
        sep = '\n'
        data = f"{sep.join(str(metadata) for metadata in storage)}{sep}"
        storage_fp.write(data)


def _create_filename(numerical_parameters: NumericalParameters, 
                     alpha_range: Iterable| float, 
                     theta_range: Iterable| float, 
                     experiment_name: str) -> str:
    h0, epsilon = numerical_parameters
    try:
        alpha_left = alpha_range[0]
        alpha_right = alpha_range[-1] if (len(alpha_range) > 1) else None
    except TypeError:
        alpha_left = alpha_range
        alpha_right = None
    try:    
        theta_left = theta_range[0]
        theta_right = theta_range[-1] if (len(theta_range) > 1) else None
    except TypeError:
        theta_left = theta_range
        theta_right = None
    base_name = "storage"
    alpha_suffix = f"alpha_from_{alpha_left}_to_{alpha_right}" if (alpha_right is not None) else f"alpha_{alpha_left}"
    theta_suffix = f"theta_from_{theta_left}_to_{theta_right}" if (theta_right is not None) else f"theta_{theta_left}"
    numerical_parameters_suffix = f"h0_{h0}_epsilon_{epsilon}"
    return '_'.join((base_name, experiment_name, numerical_parameters_suffix, alpha_suffix, theta_suffix,)) + ".data"


def perform_zero_singular_experiment(numerical_problem: NumericalProblem,
                                     alpha_range: Iterable| float, 
                                     theta_range: Iterable| float, 
                                     num_mu_range: Iterable, 
                                     show_progress: bool,
                                     N_num_mu_points: int = 256) -> None:
    numerical_parameters = numerical_problem.numerical_parameters
    experiment_name = "zero_singular"
    storage_filename = _create_filename(numerical_parameters, alpha_range, theta_range, experiment_name)
    gen =  zs.getting_statistics(numerical_problem, theta_range, alpha_range, num_mu_range, N_num_mu_points, show_progress)
    storage = []
    count = 1
    try:
        for metadata in gen:
            storage.append(metadata)
            if (count % 16) == 0:
                _save_storage(storage, storage_filename)
                storage = []
            count += 1
    except KeyboardInterrupt:
        if storage:
            _save_storage(storage, storage_filename)
        raise 


def perform_newton_experiment(numerical_problem: NumericalProblem,
                              alpha_range: Iterable| float, 
                              theta_range: Iterable| float, 
                              num_mu_initial: float, 
                              show_progress: bool) -> None:
    numerical_parameters = numerical_problem.numerical_parameters
    experiment_name = "newton"
    storage_filename = _create_filename(numerical_parameters, alpha_range, theta_range, experiment_name)
    gen =  newt.getting_statistics(numerical_problem, theta_range, alpha_range, num_mu_initial, show_progress)
    storage = []
    count = 1
    try:
        for metadata in gen:
            storage.append(metadata)
            if (count % 16) == 0:
                _save_storage(storage, storage_filename)
                storage = []
            count += 1
    except KeyboardInterrupt:
        if storage:
            _save_storage(storage, storage_filename)
        raise 


def main():
    cli_args = parse_cli()
    subcommand = cli_args.subcommand
    if subcommand == "symbolic":
        create_experiment(h0 = 1e-2, epsilon = 0.05)
    elif subcommand == "zero_singular":
        alpha_range, theta_range, num_mu_range, numerical_problem, show_progress = initialize_numerical_experiment(cli_args)
        perform_zero_singular_experiment(numerical_problem, alpha_range, theta_range, num_mu_range, show_progress)
    elif subcommand == "newton":
        alpha_range, theta_range, num_mu_range, numerical_problem, show_progress = initialize_numerical_experiment(cli_args)
        num_mu_initial = sum(num_mu_range) / 2
        perform_newton_experiment(numerical_problem, alpha_range, theta_range, num_mu_initial, show_progress)
    else:
        raise NotImplemented(f"Unknown command \"{subcommand}\"")

if __name__ == "__main__":
    main()
    
