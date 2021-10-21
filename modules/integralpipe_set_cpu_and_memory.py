import json

import kfp.dsl as _kfp_dsl
import kfp.components as _kfp_components

from collections import OrderedDict
from kubernetes import client as k8s_client


def integral1(density_p: int):
    _kale_pipeline_parameters_block = '''
    density_p = {}
    '''.format(density_p)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import numpy as np
    import math
    '''

    _kale_block2 = '''
    def compute_error(obj,approx):
        \'\'\'
        Relative or absolute error between obj and approx.
        \'\'\'
        if math.fabs(obj) > np.nextafter(0,1):
            Err = math.fabs(obj-approx)/math.fabs(obj)
        else:
            Err = math.fabs(obj-approx)
        return Err
    '''

    _kale_block3 = '''
    f = lambda x: 4/(1+x**2)
    x_p = np.random.uniform(0,1,density_p)
    obj = math.pi
    a = 0
    b = 1
    vol = b-a
    ex_1 = vol*np.mean(f(x_p))
    print("error relativo: {:0.4e}".format(compute_error(obj, ex_1)))
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_block3,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/integral1.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('integral1')

    _kale_mlmdutils.call("mark_execution_complete")


def integral2(density_p: int):
    _kale_pipeline_parameters_block = '''
    density_p = {}
    '''.format(density_p)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import numpy as np
    import math
    '''

    _kale_block2 = '''
    def compute_error(obj,approx):
        \'\'\'
        Relative or absolute error between obj and approx.
        \'\'\'
        if math.fabs(obj) > np.nextafter(0,1):
            Err = math.fabs(obj-approx)/math.fabs(obj)
        else:
            Err = math.fabs(obj-approx)
        return Err
    '''

    _kale_block3 = '''
    f = lambda x: 1/x
    x_p = np.random.uniform(1,2,density_p)
    obj = math.log(2)
    a = 1
    b = 2
    vol = b-a
    ex_2 = vol*np.mean(f(x_p))
    print("error relativo: {:0.4e}".format(compute_error(obj, ex_2)))
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_block3,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/integral2.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('integral2')

    _kale_mlmdutils.call("mark_execution_complete")


def integral3(density_p: int):
    _kale_pipeline_parameters_block = '''
    density_p = {}
    '''.format(density_p)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import numpy as np
    import math
    '''

    _kale_block2 = '''
    def compute_error(obj,approx):
        \'\'\'
        Relative or absolute error between obj and approx.
        \'\'\'
        if math.fabs(obj) > np.nextafter(0,1):
            Err = math.fabs(obj-approx)/math.fabs(obj)
        else:
            Err = math.fabs(obj-approx)
        return Err
    '''

    _kale_block3 = '''
    f = lambda x,y:x**2+y**2
    a1 = -1
    b1 = 1
    a2 = 0
    b2 = 1
    x_p = np.random.uniform(a1,b1,density_p)
    y_p = np.random.uniform(a2,b2,density_p)
    obj = 4/3
    vol = (b1-a1)*(b2-a2)
    ex_3 = vol*np.mean(f(x_p,y_p))
    print("error relativo: {:0.4e}".format(compute_error(obj, ex_3)))
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_block3,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/integral3.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('integral3')

    _kale_mlmdutils.call("mark_execution_complete")


def integral4(density_p: int):
    _kale_pipeline_parameters_block = '''
    density_p = {}
    '''.format(density_p)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import numpy as np
    import math
    '''

    _kale_block2 = '''
    def compute_error(obj,approx):
        \'\'\'
        Relative or absolute error between obj and approx.
        \'\'\'
        if math.fabs(obj) > np.nextafter(0,1):
            Err = math.fabs(obj-approx)/math.fabs(obj)
        else:
            Err = math.fabs(obj-approx)
        return Err
    '''

    _kale_block3 = '''
    f = lambda x,y:np.cos(x)*np.sin(y)
    a1 = 0
    b1 = math.pi/2
    a2 = 0
    b2 = math.pi/2
    x_p = np.random.uniform(a1,b1,density_p)
    y_p = np.random.uniform(a2,b2,density_p)
    obj = 1
    vol = (b1-a1)*(b2-a2)
    ex_4 = vol*np.mean(f(x_p,y_p))
    print("error relativo: {:0.4e}".format(compute_error(obj, ex_4)))
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_block3,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/integral4.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('integral4')

    _kale_mlmdutils.call("mark_execution_complete")


def integral5(density_p: int):
    _kale_pipeline_parameters_block = '''
    density_p = {}
    '''.format(density_p)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import numpy as np
    import math
    '''

    _kale_block2 = '''
    def compute_error(obj,approx):
        \'\'\'
        Relative or absolute error between obj and approx.
        \'\'\'
        if math.fabs(obj) > np.nextafter(0,1):
            Err = math.fabs(obj-approx)/math.fabs(obj)
        else:
            Err = math.fabs(obj-approx)
        return Err
    '''

    _kale_block3 = '''
    f = lambda x,y,z:(x+2*y+3*z)**2
    a1 = 0
    b1 = 1
    a2 = -1/2
    b2 = 0
    a3 = 0
    b3 = 1/3
    x_p = np.random.uniform(a1,b1,density_p)
    y_p = np.random.uniform(a2,b2,density_p)
    z_p = np.random.uniform(a3,b3,density_p)
    obj = 1/12
    vol = (b1-a1)*(b2-a2)*(b3-a3)
    ex_5 = vol*np.mean(f(x_p,y_p,z_p))
    print("error relativo: {:0.4e}".format(compute_error(obj, ex_5)))
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_block3,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/integral5.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('integral5')

    _kale_mlmdutils.call("mark_execution_complete")


def integral6(density_p: int):
    _kale_pipeline_parameters_block = '''
    density_p = {}
    '''.format(density_p)

    from kale.common import mlmdutils as _kale_mlmdutils
    _kale_mlmdutils.init_metadata()

    _kale_block1 = '''
    import numpy as np
    import math
    '''

    _kale_block2 = '''
    def compute_error(obj,approx):
        \'\'\'
        Relative or absolute error between obj and approx.
        \'\'\'
        if math.fabs(obj) > np.nextafter(0,1):
            Err = math.fabs(obj-approx)/math.fabs(obj)
        else:
            Err = math.fabs(obj-approx)
        return Err
    '''

    _kale_block3 = '''
    f = lambda x: 4/(1+x**2)
    x_p = np.random.uniform(0,1,density_p)
    obj = math.pi
    a = 0
    b = 1
    vol = b-a
    f_bar = np.mean(f(x_p))
    ex_6 = vol*f_bar
    print("error relativo: {:0.4e}".format(compute_error(obj,ex_6 )))
    '''

    # run the code blocks inside a jupyter kernel
    from kale.common.jputils import run_code as _kale_run_code
    from kale.common.kfputils import \
        update_uimetadata as _kale_update_uimetadata
    _kale_blocks = (_kale_pipeline_parameters_block,
                    _kale_block1,
                    _kale_block2,
                    _kale_block3,
                    )
    _kale_html_artifact = _kale_run_code(_kale_blocks)
    with open("/integral6.html", "w") as f:
        f.write(_kale_html_artifact)
    _kale_update_uimetadata('integral6')

    _kale_mlmdutils.call("mark_execution_complete")


_kale_integral1_op = _kfp_components.func_to_container_op(
    integral1, base_image='cdasitam/example-cdas-project-kale:0.6.1')


_kale_integral2_op = _kfp_components.func_to_container_op(
    integral2, base_image='cdasitam/example-cdas-project-kale:0.6.1')


_kale_integral3_op = _kfp_components.func_to_container_op(
    integral3, base_image='cdasitam/example-cdas-project-kale:0.6.1')


_kale_integral4_op = _kfp_components.func_to_container_op(
    integral4, base_image='cdasitam/example-cdas-project-kale:0.6.1')


_kale_integral5_op = _kfp_components.func_to_container_op(
    integral5, base_image='cdasitam/example-cdas-project-kale:0.6.1')


_kale_integral6_op = _kfp_components.func_to_container_op(
    integral6, base_image='cdasitam/example-cdas-project-kale:0.6.1')


@_kfp_dsl.pipeline(
    name='integralpipe-pr7xy',
    description='Example compute integrals in pipeline'
)
def auto_generated_pipeline(density_p='1000000000', vol_shared_volume='hostpath-pvc'):
    _kale_pvolumes_dict = OrderedDict()
    _kale_volume_step_names = []
    _kale_volume_name_parameters = []

    _kale_annotations = {}

    _kale_volume = _kfp_dsl.PipelineVolume(pvc=vol_shared_volume)

    _kale_pvolumes_dict['/shared_volume'] = _kale_volume

    _kale_volume_step_names.sort()
    _kale_volume_name_parameters.sort()

    _kale_integral1_task = _kale_integral1_op(density_p)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()
    
    #see: https://kubeflow-pipelines.readthedocs.io/en/latest/source/kfp.dsl.html#kfp.dsl.Sidecar.add_resource_limit
    #can also be: _kale_step_limits = {'nvidia.com/gpu': '1'}
    _kale_step_limits = {'cpu': '1',
                         'memory': '40Gi'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_integral1_task.container.add_resource_limit(_kale_k, _kale_v)
        
    _kale_integral1_task.container.working_dir = "//shared_volume"
    _kale_integral1_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'integral1': '/integral1.html'})
    _kale_integral1_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_integral1_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_integral1_task.dependent_names +
                       _kale_volume_step_names)
    _kale_integral1_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_integral1_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_integral2_task = _kale_integral2_op(density_p)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()

    _kale_step_limits = {'cpu': '1',
                         'memory': '40Gi'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_integral2_task.container.add_resource_limit(_kale_k, _kale_v)
        
    _kale_integral2_task.container.working_dir = "//shared_volume"
    _kale_integral2_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'integral2': '/integral2.html'})
    _kale_integral2_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_integral2_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_integral2_task.dependent_names +
                       _kale_volume_step_names)
    _kale_integral2_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_integral2_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))
        
    _kale_integral3_task = _kale_integral3_op(density_p)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()

    _kale_step_limits = {'cpu': '1',
                         'memory': '40Gi'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_integral3_task.container.add_resource_limit(_kale_k, _kale_v)
        
    _kale_integral3_task.container.working_dir = "//shared_volume"
    _kale_integral3_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'integral3': '/integral3.html'})
    _kale_integral3_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_integral3_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_integral3_task.dependent_names +
                       _kale_volume_step_names)
    _kale_integral3_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_integral3_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))
        
    _kale_integral4_task = _kale_integral4_op(density_p)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()
    
    _kale_step_limits = {'cpu': '1',
                         'memory': '40Gi'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_integral4_task.container.add_resource_limit(_kale_k, _kale_v)
        
    _kale_integral4_task.container.working_dir = "//shared_volume"
    _kale_integral4_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'integral4': '/integral4.html'})
    _kale_integral4_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_integral4_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_integral4_task.dependent_names +
                       _kale_volume_step_names)
    _kale_integral4_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_integral4_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_integral5_task = _kale_integral5_op(density_p)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()
    
    _kale_step_limits = {'cpu': '1',
                         'memory': '40Gi'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_integral5_task.container.add_resource_limit(_kale_k, _kale_v)
        
    _kale_integral5_task.container.working_dir = "//shared_volume"
    _kale_integral5_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'integral5': '/integral5.html'})
    _kale_integral5_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_integral5_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_integral5_task.dependent_names +
                       _kale_volume_step_names)
    _kale_integral5_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_integral5_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))

    _kale_integral6_task = _kale_integral6_op(density_p)\
        .add_pvolumes(_kale_pvolumes_dict)\
        .after()
    
    _kale_step_limits = {'cpu': '1',
                         'memory': '40Gi'}
    for _kale_k, _kale_v in _kale_step_limits.items():
        _kale_integral6_task.container.add_resource_limit(_kale_k, _kale_v)  
        
    _kale_integral6_task.container.working_dir = "//shared_volume"
    _kale_integral6_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    _kale_output_artifacts = {}
    _kale_output_artifacts.update(
        {'mlpipeline-ui-metadata': '/tmp/mlpipeline-ui-metadata.json'})
    _kale_output_artifacts.update({'integral6': '/integral6.html'})
    _kale_integral6_task.output_artifact_paths.update(_kale_output_artifacts)
    _kale_integral6_task.add_pod_label(
        "pipelines.kubeflow.org/metadata_written", "true")
    _kale_dep_names = (_kale_integral6_task.dependent_names +
                       _kale_volume_step_names)
    _kale_integral6_task.add_pod_annotation(
        "kubeflow-kale.org/dependent-templates", json.dumps(_kale_dep_names))
    if _kale_volume_name_parameters:
        _kale_integral6_task.add_pod_annotation(
            "kubeflow-kale.org/volume-name-parameters",
            json.dumps(_kale_volume_name_parameters))


if __name__ == "__main__":
    pipeline_func = auto_generated_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Get or create an experiment and submit a pipeline run
    import kfp
    client = kfp.Client()
    experiment = client.create_experiment('integralexp')

    # Submit a pipeline run
    from kale.common.kfputils import generate_run_name
    run_name = generate_run_name('integralpipe-pr7xy')
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
