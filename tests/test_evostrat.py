"""
Some unittesting code adopted from Lisandro Dalcin's MPI4PY project as
evostrat requires testing with MPI4PY
"""
import unittest
import sys, os, shlex
import subprocess as sp
import mpi4py


def find_executable(exe):
    from distutils.spawn import find_executable as find_exe
    command = shlex.split(exe)
    executable = find_exe(command[0])
    if executable:
        command[0] = executable
        return ' '.join(command)


def find_mpiexec(mpiexec='mpiexec'):
    mpiexec = os.environ.get('MPIEXEC') or mpiexec
    mpiexec = find_executable(mpiexec)
    if not mpiexec and sys.platform.startswith('win'):
        MSMPI_BIN = os.environ.get('MSMPI_BIN', '')
        mpiexec = os.path.join(MSMPI_BIN, mpiexec)
        mpiexec = find_executable(mpiexec)
    if not mpiexec:
        mpiexec = find_executable('mpirun')
    return mpiexec


def launcher(num_proc, script):
    mpiexec = find_mpiexec()
    python = sys.executable
    command = '{mpiexec} -n {num_proc} {python} {script}'
    return shlex.split(command.format(**vars()))


def execute(num_proc, script):
    env = os.environ.copy()
    pypath = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    pypath.insert(0, os.path.abspath(os.path.dirname(mpi4py.__path__[0])))
    env['PYTHONPATH'] = os.pathsep.join(pypath)
    cmdline = launcher(num_proc, script)
    p = sp.Popen(cmdline, stdout=sp.PIPE, stderr=sp.PIPE, env=env, bufsize=0)
    stdout, stderr = p.communicate()
    return p.returncode, stdout.decode(), stderr.decode()


class TestES(unittest.TestCase):
    pyfile = 'es_test_script.py'

    def test_basic_es(self):
        dirname = os.path.abspath(os.path.dirname(__file__))
        script = os.path.join(dirname, self.pyfile)
        status, stdout, stderr = execute(4, script)
        self.assertEqual(status, 0)
        self.assertEqual(stderr, '')


class TestGA(unittest.TestCase):
    pyfile = 'ga_test_script.py'

    def test_basic_es(self):
        dirname = os.path.abspath(os.path.dirname(__file__))
        script = os.path.join(dirname, self.pyfile)
        status, stdout, stderr = execute(4, script)
        self.assertEqual(status, 0)
        self.assertEqual(stderr, '')


if __name__ == '__main__':
    unittest.main()