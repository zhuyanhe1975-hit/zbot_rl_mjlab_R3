"""Native per-world weld-load checks for the actual mjbatch states."""
import ctypes
import hashlib
import os
import subprocess
from pathlib import Path
import mujoco
import numpy as np
from .model import ROOT

class Monitor:
    def __init__(self,model,threads):
        source=ROOT/'src/zbot_mjbatch/native/monitor.cpp'
        header=Path.home()/'Github/mjbatch/src/mjbatch/csrc/threadpool.h'
        if not header.is_file():raise FileNotFoundError(f'mjbatch source header not found: {header}')
        cache=ROOT/'outputs/reconfiguration/mjbatch_rl/native';cache.mkdir(parents=True,exist_ok=True)
        mjdir=Path(mujoco.__file__).resolve().parent
        libs=sorted(mjdir.glob('libmujoco.so.*'))
        if not libs:raise RuntimeError('MuJoCo shared library not found')
        key=hashlib.sha256(source.read_bytes()+header.read_bytes()+mujoco.__version__.encode()).hexdigest()[:16]
        libfile=cache/f'monitor_{key}.so'
        if not libfile.exists():
            tmp=cache/f'monitor_{key}.{os.getpid()}.tmp.so'
            subprocess.run(['g++','-std=c++17','-O3','-fPIC','-shared','-pthread',f'-I{mjdir/"include"}',
                            f'-I{header.parent}',str(source),f'-L{mjdir}',f'-l:{libs[-1].name}',
                            f'-Wl,-rpath,{mjdir}','-o',str(tmp)],check=True)
            tmp.replace(libfile)
        modelpath=cache/f'model_{key}.{os.getpid()}.{id(self)}.mjb';mujoco.mj_saveModel(model,str(modelpath),None)
        self.lib=ctypes.CDLL(str(libfile));self.lib.zm_create.argtypes=[ctypes.c_char_p,ctypes.c_int];self.lib.zm_create.restype=ctypes.c_void_p
        self.lib.zm_free.argtypes=[ctypes.c_void_p]
        arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.zm_measure.argtypes=[ctypes.c_void_p,arr,ctypes.c_int,arr]
        try:self.ptr=self.lib.zm_create(str(modelpath).encode(),threads)
        finally:modelpath.unlink(missing_ok=True)
        if not self.ptr:raise RuntimeError('Could not initialize native safety monitor')
    def measure(self,states,result):self.lib.zm_measure(self.ptr,states,len(states),result)
    def close(self):
        if self.ptr:self.lib.zm_free(self.ptr);self.ptr=None
