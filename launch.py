import mpi4py.MPI as MPI
import yaml
import logging
import os
import shutil
import time
import config
import numpy as np
import tensorflow as tf
import tb_logger

from experiment import Experiment
from config import expand_dot_items, flatten_dot_items, DotDict, GLOBAL_CONFIG


def _warn_new_keys(config, existing_config):
    for k in config.keys() - existing_config.keys():
        if '.mpi_' not in k:
            logging.warning(f'Specified config key {k} does not exist.')


def _merge_config(config, config_files, task_config=None):
    derived_config = {}
    config_files = ['configs/default.yaml'] + config_files
    for cfg in config_files:
        with open(cfg, mode='r') as f:
            new_config = flatten_dot_items(yaml.safe_load(f))
            if len(derived_config) > 0:
                _warn_new_keys(new_config, derived_config)
            derived_config.update(new_config)
    if task_config is not None:
        _warn_new_keys(task_config, derived_config)
        derived_config.update(task_config)
    if config is not None:
        _warn_new_keys(config, derived_config)
        derived_config.update(config)
    derived_config = flatten_dot_items(derived_config)
    return derived_config


def _create_run_name(tags):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if tags:
        clean_tags = [t.replace(os.sep, '_').replace(' ', '_') for t in tags[:3]]
        tag_suffix = "-".join(clean_tags)
        return f"{timestamp}-{tag_suffix}"
    return timestamp


def _prepare_log_dir(base_dir, tags):
    log_root = base_dir or 'runs'
    run_name = _create_run_name(tags)
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _save_slurm_metadata(log_dir):
    if 'SLURM_JOB_ID' not in os.environ:
        return
    os.makedirs(log_dir, exist_ok=True)
    if 'SLURM_ARRAY_JOB_ID' in os.environ:
        job_id = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    else:
        job_id = os.environ['SLURM_JOB_ID']
    meta_path = os.path.join(log_dir, 'slurm_job.txt')
    with open(meta_path, mode='w') as f:
        f.write(f"{job_id}\n")
    log_src = f'slurm-{job_id}.out'
    if os.path.exists(log_src):
        log_dst = os.path.join(log_dir, f'slurm-{job_id}.txt')
        if not os.path.exists(log_dst):
            try:
                os.link('./' + log_src, log_dst)
            except OSError:
                shutil.copy(log_src, log_dst)


def _sync_config(comm, mpi_rank, config):
    return comm.bcast(config if mpi_rank == 0 else None, root=0)


def _update_ranked_config(config: dict, mpi_rank: int):
    # TODO potentially add support for dictionaries within the mpi_split
    for k, v in filter(lambda it: 'mpi_split' in it[0], list(config.items())):
        base_key = k.replace('.mpi_split', '')
        repeat_key = f'{base_key}.mpi_repeat'
        repeat = config.pop(repeat_key, 1)
        idx = mpi_rank // repeat
        selected_option = v[idx % len(v)]
        config[base_key] = selected_option
        del config[k]
    return config


def _create_array_task(spec, mpi_rank, array_subset):
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    tasks = spec['array']
    if array_subset is not None and len(array_subset) > 0:
        task = tasks[array_subset[task_id - 1]]
    else:
        task = tasks[task_id - 1]
    if mpi_rank == 0:
        logging.info(f'Loading task {task_id}:\n{yaml.dump(task)}')
    tags = task.get('tags', [])
    config_files = task.get('config_files', [])
    config = task.get('config', {})
    config['task_id'] = task_id
    return tags, config_files, config


def _create_grid_task(spec, mpi_rank):
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    grid = spec['grid']
    task_count = np.prod([len(ax) for ax in grid])
    if task_id > task_count:
        raise ValueError(f'There are only {task_count} tasks, {task_id} was requested')
    selection = []
    i = task_id - 1  # One based task id
    for ax in grid:
        selection.append(ax[i % len(ax)])
        i //= len(ax)

    if mpi_rank == 0:
        logging.info(f'Loading grid selection {task_id} of {task_count}:\n{yaml.dump(selection)}')

    tags = []
    config_files = []
    config = dict(task_id=task_id)
    for ax in selection:
        tags.extend(ax.get('tags', []))
        config_files.extend(ax.get('config_files', []))
        config.update(flatten_dot_items(ax.get('config', {})))
    return tags, config_files, config


def _setup_config(mpi_rank, config, config_files, array_file, array_subset):
    tags = []
    config_files = config_files or []
    if array_file is not None:
        with open(array_file, mode='r') as f:
            spec = yaml.safe_load(f)
            if 'array' in spec:
                t_tags, t_config_files, t_config = _create_array_task(spec, mpi_rank, array_subset)
            elif 'grid' in spec:
                t_tags, t_config_files, t_config = _create_grid_task(spec, mpi_rank)
            tags.extend(t_tags)
            config_files.extend(t_config_files)
            task_config = t_config
    else:
        task_config = None
    config = _merge_config(config, config_files, task_config)
    return config, tags


def run(args):
    log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
    tf_log_level = os.environ.get('TF_LOGLEVEL', 'WARN').upper()
    logging.basicConfig(level=log_level)
    tf.get_logger().setLevel(tf_log_level)
    logging.info('Launching')

    # Disable tensorflow GPU support
    tf.config.experimental.set_visible_devices([], "GPU")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    config = None
    tags = []
    log_dir = None
    if rank == 0:
        config, tags = _setup_config(rank, args.config, args.config_files, args.array, args.subset)
        tags = tags + args.tags if args.tags else tags
        config['mpi_size'] = comm.Get_size()
        log_dir = _prepare_log_dir(args.logdir, tags)
    config = _sync_config(comm, rank, config)
    log_dir = comm.bcast(log_dir, root=0)
    config = _update_ranked_config(config, rank)
    config = expand_dot_items(DotDict(config))
    config.log_dir = log_dir
    GLOBAL_CONFIG.update(config)
    tb_logger.init(log_dir, enabled=rank == 0)
    if rank == 0:
        tb_logger.save_config(config, tags)
        _save_slurm_metadata(log_dir)
    experiment = Experiment(config)
    entry_fn = getattr(experiment, config.call)
    entry_fn()


if __name__ == '__main__':
    run(config.parse_args())
