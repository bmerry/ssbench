# ssbench: benchmarking of sort and scan libraries
# Copyright (C) 2014  Bruce Merry
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os.path
from waflib import Task
from waflib.Tools import ccroot, c_preproc, cxx
from waflib.TaskGen import extension, feature, after_method
from waflib.Configure import conf

top = '.'
out = 'build'

class cuda(Task.Task):
    run_str = '${NVCC} ${CUDAFLAGS} ${CPPPATH_ST:INCPATHS} ${DEFINES_ST:DEFINES} -c ${SRC} -o ${TGT}'
    ext_in = '.cu'
    ext_out = '.o'
    color = 'GREEN'
    vars = ['CCDEPS']
    scan = c_preproc.scan
    shell = False

@extension('.cu')
def c_hook(self, node):
    return self.create_compiled_task('cuda', node)

@feature('cuda')
@after_method('propagate_uselib_vars', 'process_source')
def apply_incpaths(self):
    lst = self.to_incnodes(self.to_list(getattr(self, 'includes', [])) + self.env['INCLUDES'])
    self.includes_nodes = lst
    self.env['INCPATHS'] = [x.abspath() for x in lst]

ccroot.USELIB_VARS['cuda'] = ccroot.USELIB_VARS['cxx']
ccroot.USELIB_VARS['cudaprogram'] = ccroot.USELIB_VARS['cxxprogram']

class cudaprogram(cxx.cxxprogram):
    run_str = '${NVCC} ${CUDAFLAGS} ${NVCC_XCOMPILER:LINKFLAGS} ${CXXLNK_SRC_F}${SRC} ${CXXLNK_TGT_F}${TGT[0].abspath()} ${RPATH_ST:RPATH} ${FRAMEWORKPATH_ST:FRAMEWORKPATH} ${FRAMEWORK_ST:FRAMEWORK} ${ARCH_ST:ARCH} ${STLIBPATH_ST:STLIBPATH} ${STLIB_ST:STLIB} ${LIBPATH_ST:LIBPATH} ${LIB_ST:LIB}'

@conf
def check_cuda(self, *args, **kw):
    kw['features'] = ['cuda', 'cudaprogram']
    kw['compile_filename'] = 'test.cu'
    return self.check(*args, **kw)

def options(ctx):
    ctx.load('compiler_cxx')
    ctx.add_option('--with-clogs', action = 'store', help = 'Path to CLOGS')
    ctx.add_option('--with-compute', action = 'store', help = 'Path to Boost.Compute')
    ctx.add_option('--with-vexcl', action = 'store', help = 'Path to VexCL')
    ctx.add_option('--with-bolt', action = 'store', help = 'Path to Bolt')
    ctx.add_option('--with-thrust', action = 'store', help = 'Path to Thrust')
    ctx.add_option('--with-cub', action = 'store', help = 'Path to CUB')
    ctx.add_option('--with-mgpu', action = 'store', help = 'Path to ModernGPU')

@conf
def check_library(self, func, option, includes_add, libpath_add, *args, **kw):
    if option is not None:
        if option:
            kw['includes'] = [os.path.join(option, x) for x in includes_add]
            kw['libpath'] = [os.path.join(option, x) for x in libpath_add]
        return func(*args, **kw)
    else:
        return func(*args, mandatory = False, **kw)

@conf
def check_cuda_library(self, func, option, includes_add, libpath_add, *args, **kw):
    if option is not None:
        self.find_program('nvcc', var = 'NVCC')
        return self.check_library(func, option, includes_add, libpath_add, *args, **kw)
    else:
        if not self.find_program('nvcc', var = 'NVCC', mandatory = False):
            return False
        return func(*args, mandatory = False, **kw)

def configure(ctx):
    ctx.load('compiler_cxx')

    ctx.add_os_flags('NVCC')
    ctx.add_os_flags('CUDAFLAGS')
    ctx.env.append_value('CXXFLAGS', ['-std=c++11', '-fopenmp'])
    ctx.env.append_value('LINKFLAGS', ['-fopenmp'])
    for arch, code in [('20', '20'), ('20', '21'), ('30', '30'), ('32', '32'), ('35', '35')]:
        ctx.env.append_value('CUDAFLAGS', ['-gencode', 'arch=compute_{},code=sm_{}'.format(arch, code)])

    ctx.env.have_thrust = ctx.check_cuda_library(ctx.check_cuda, ctx.options.with_thrust, [''], [],
        header_name = 'thrust/scan.h', uselib_store = 'THRUST')
    ctx.env.have_cub = ctx.check_cuda_library(ctx.check_cuda, ctx.options.with_cub, [''], [],
        header_name = 'cub/cub.cuh', uselib_store = 'CUB')
    ctx.env.have_mgpu = ctx.check_cuda_library(ctx.check_cuda, ctx.options.with_mgpu, ['include'], [],
        header_name = 'moderngpu.cuh', uselib_store = 'MGPU')
    ctx.env.mgpu_path = ctx.options.with_mgpu
    ctx.env.have_clogs = ctx.check_library(ctx.check_cxx, ctx.options.with_clogs, ['include'], ['lib'],
        header_name = 'clogs/clogs.h', lib = ['clogs', 'OpenCL'], uselib_store = 'CLOGS')
    ctx.env.have_compute = ctx.check_library(ctx.check_cxx, ctx.options.with_compute, ['include'], [],
        header_name = 'boost/compute.hpp', lib = ['OpenCL'], uselib_store = 'COMPUTE')
    ctx.env.have_vexcl = ctx.check_library(ctx.check_cxx, ctx.options.with_vexcl, [''], [],
        header_name = 'vexcl/vexcl.hpp', lib = ['boost_system', 'OpenCL'], uselib_store = 'VEXCL')
    ctx.env.have_bolt = ctx.check_library(ctx.check_cxx, ctx.options.with_bolt, ['include', 'build/include'], ['build/bolt/cl'],
        header_name = 'bolt/cl/bolt.h', stlib = ['clBolt.runtime.gcc'],
        lib = ['boost_system', 'OpenCL'], uselib_store = 'BOLT')

    ctx.check_cxx(
        header_name = 'boost/program_options.hpp',
        lib = 'boost_program_options',
        uselib_store = 'PROGRAM_OPTIONS')

def build(ctx):
    sources = ['ssbench.cpp', 'cpu.cpp', 'clutils.cpp', 'hostutils.cpp']
    use = ['PROGRAM_OPTIONS']
    need_cuda = False
    features = ['cxx']

    ctx.env.append_value('CXXFLAGS', ['-Wall', '-O3'])
    ctx.env.append_value('CUDAFLAGS', ['-Xcompiler=-Wall', '-O3'])

    ctx.env['NVCC_XCOMPILER'] = '-Xcompiler=%s'
    ctx.env.append_value('DEFINES', ['__CL_ENABLE_EXCEPTIONS', 'BOOST_DISABLE_ASSERTS'])

    if ctx.env.have_clogs:
        sources += ['clogs.cpp']
        use += ['CLOGS']
    if ctx.env.have_compute:
        sources += ['compute.cpp']
        use += ['COMPUTE']
    if ctx.env.have_vexcl:
        sources += ['vex.cpp']
        use += ['VEXCL']
    if ctx.env.have_bolt:
        sources += ['bolt.cpp']
        use += ['BOLT']

    if ctx.env.have_thrust:
        sources += ['thrust.cu']
        use += ['THRUST']
        need_cuda = True
    if ctx.env.have_cub:
        sources += ['cub.cu']
        use += ['CUB']
        need_cuda = True
    if ctx.env.have_mgpu:
        mgpuutil = ctx(
                features = ['cuda', 'cxx'],
                target = 'mgpuutil_o',
                source = ctx.root.find_resource(os.path.join(ctx.env.mgpu_path, 'src', 'mgpuutil.cpp')),
                use = ['MGPU'])
        mgpuutil.mappings['.cpp'] = c_hook
        sources += [
            'mgpu.cu',
            ctx.root.find_resource(os.path.join(ctx.env.mgpu_path, 'src', 'mgpucontext.cu')),
        ]
        use += ['MGPU', 'mgpuutil_o']
        need_cuda = True

    if need_cuda:
        features += ['cuda', 'cudaprogram']
    else:
        features += ['cxxprogram']

    ctx(features = features, source = sources, target = 'ssbench', use = use)
