*Notebooks* de referencia que tienen celdas identificadas con [kale](https://github.com/kubeflow-kale/kale). Si los ejecutan con el panel de *kale* en su infraestructura de [minikube](https://github.com/kubernetes/minikube) y [kubeflow](https://github.com/kubeflow/kubeflow) deben terminar exitosamente sin cambios :)

Ver [6.Minikube-y-AWS](https://github.com/ITAM-DS/analisis-numerico-computo-cientifico/wiki/6.Minikube-y-AWS) para uso de *minikube* en AWS.

Algunas ejecuciones que ayudan desde la línea de comandos al envío hacia `kubeflow`

```
sudo kale --nb titanic_dataset_ml.ipynb --experiment_name default --pipeline_name titanicml --kfp_host http://<master DNS cluster kubernetes>:<port> --upload_pipeline --debug

sudo kale --nb titanic_dataset_ml.ipynb --experiment_name default --pipeline_name titanicml --debug


sudo kale --nb titanic_dataset_ml.ipynb --experiment_name default --pipeline_name titanicml --kfp_host <host>:<port> --upload_pipeline --debug
```
