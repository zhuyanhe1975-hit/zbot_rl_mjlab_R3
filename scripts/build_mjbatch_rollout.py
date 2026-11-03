"""Build the native MuJoCo rollout bridge using mjbatch's ThreadPool header."""
import argparse
import subprocess
from pathlib import Path
import mujoco


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mjbatch-root',type=Path,default=Path.home()/'Github/mjbatch')
    p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/mjbatch_refinement/libcurl_rollout.so'))
    p.add_argument('--source',type=Path,default=Path('scripts/mjbatch_curl_rollout.cpp'))
    a=p.parse_args()
    mujoco_dir=Path(mujoco.__file__).resolve().parent
    libraries=sorted(mujoco_dir.glob('libmujoco.so.*'))
    header=a.mjbatch_root/'src/mjbatch/csrc/threadpool.h'
    if not libraries or not header.is_file():p.error('MuJoCo shared library or mjbatch threadpool.h not found')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    cmd=['g++','-std=c++17','-O3','-fPIC','-shared','-pthread',f'-I{mujoco_dir / "include"}',
         f'-I{header.parent}',str(a.source),f'-L{mujoco_dir}',
         f'-l:{libraries[-1].name}',f'-Wl,-rpath,{mujoco_dir}','-o',str(a.output)]
    subprocess.run(cmd,check=True);print('Built',a.output,'using MuJoCo',mujoco.__version__,'and',header)

if __name__=='__main__':main()
