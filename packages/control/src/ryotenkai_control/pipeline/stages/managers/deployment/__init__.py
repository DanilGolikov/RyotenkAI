"""Components composed by ``TrainingDeploymentManager`` (Wave 3 decomposition).

Each component owns one concern of GPU training deployment:

* :mod:`code_syncer` — rsync/tar-pipe of required source modules to remote.
* (future) ``file_uploader`` — config + dataset + secrets upload via SCP/tar.
* (future) ``dependency_installer`` — Docker image presence + runtime check.
* (future) ``training_launcher`` — env file creation + cloud/Docker spawn.

Shared helpers live in :mod:`ssh_helpers` and (later) ``provider_config``.
"""
