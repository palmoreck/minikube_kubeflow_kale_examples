
Use in a instance with more than 40 GB (for example `m5.4xlarge`):

```
docker run -v /shared_volume/:/shared_volume/ -v /root/:/root/ --entrypoint=/bin/bash --rm -dit --name mycontainer cdasitam/jupyterlab-cert-kale:0.6.1 -c "python3 /shared_volume/integralpipe-set_memory.py"
```
