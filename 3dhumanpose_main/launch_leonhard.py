import os
import uuid
from pathlib import Path

import click
import colorama
import paramiko
import pathspec
from colorama import Fore as Fg, Style as St


DEFAULT_TARGET = 'kbouchiat@login.leonhard.ethz.ch'
DEFAULT_JUMP = 'kbouchiat@jumphost.inf.ethz.ch'
DEFAULT_WALLCLOCK = '04:00'

COPY_PATHSPEC = """
*

!.flake8
!.vscode/
!.mp-env/
!**/__pycache__/
"""


def resolve_default_identity() -> str:
    return Path.home() / '.ssh' / 'id_rsa'


def parse_target(target: str):
    user, host = target.split('@', maxsplit=1)
    return user, host


def sync_project(sftp: paramiko.SFTPClient, spec: pathspec.PathSpec):
    def makedirs(path):
        if path == '':
            return
        makedirs(os.path.dirname(path))
        try:
            sftp.stat(path)
        except FileNotFoundError:
            sftp.mkdir(path)

    for filename in sorted(spec.match_tree('.')):
        local_path = str(Path('.') / Path(filename))
        remote_path = '3dhumanpose_main/' + Path(filename).as_posix()

        makedirs(os.path.dirname(remote_path))
        sftp.put(local_path, remote_path)


def run_command(client: paramiko.SSHClient, command: str):
    _, stdout, stderr = client.exec_command(command)

    while True:
        line = stdout.readline()
        if not line:
            break
        print(line, end="")

    print(stderr.read().decode(), end='')
    return stdout.channel.recv_exit_status()


@click.command()
@click.option('-i', 'identity_file', type=click.Path(exists=True), default=resolve_default_identity,
              help="OpenSSH private key used for authentication. (e.g. `~/.ssh/id_rsa')")
@click.option('-T', 'targethost', default=DEFAULT_TARGET, show_default=True,
              help="Target host.")
@click.option('-J', 'jumphost', default=DEFAULT_JUMP, show_default=True,
              help="Jump host.")
@click.option('-W', 'wall_clock', default=DEFAULT_WALLCLOCK, show_default=True,
              help="Wall clock time for submitted job.")
def launch_leonhard(
    targethost: str,
    jumphost: str,
    identity_file: Path,
    wall_clock: str
):
    copyspec = pathspec.PathSpec.from_lines('gitwildmatch', COPY_PATHSPEC.splitlines())
    print(f"Using identity file `{identity_file}'.")

    user, host = parse_target(jumphost)
    jumpbox = paramiko.SSHClient()
    jumpbox.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    jumpbox.connect(host, username=user, key_filename=str(identity_file), timeout=3.0)

    user, host = parse_target(targethost)
    jumpbox_transport = jumpbox.get_transport()
    jumpbox_channel = jumpbox_transport.open_channel('direct-tcpip',
                                                     dest_addr=(host, 22),
                                                     src_addr=jumpbox_transport.getpeername())

    targetbox = paramiko.SSHClient()
    targetbox.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    targetbox.connect(host, username=user, key_filename=str(identity_file), timeout=3.0, sock=jumpbox_channel)

    print(f"Connected to `{targethost}' via `{jumphost}'.")

    sftp = targetbox.open_sftp()

    tmpdir = f'/cluster/home/{user}/mp-project-{str(uuid.uuid4())[:5]}'
    sftp.mkdir(tmpdir)
    sftp.chdir(tmpdir)

    print(f"Copying project files to `{Fg.YELLOW}{tmpdir}{St.RESET_ALL}'.")
    sync_project(sftp, copyspec)
    sftp.close()

    print("Launching job submission script.")
    run_command(targetbox, f'cd {tmpdir}/3dhumanpose_main; bash launch_self.sh {wall_clock}')

    print("Connect to Leonhard using:")
    print(f'{Fg.YELLOW}ssh -i "{identity_file}" -J {jumphost} {targethost}{St.RESET_ALL}')

    targetbox.close()
    jumpbox.close()


if __name__ == '__main__':
    colorama.init()
    launch_leonhard()
