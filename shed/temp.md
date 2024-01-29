## **Instructions to run TorchOpenFace**

* Enter build directory of OpenFace
```
cd /data/scanavan/saandeep/OpenFace2.0/TorchOpenFace
```

* Set CUDA environment variables
```
sh set_vars.sh
```

* Run sample code. This should take about 30 seconds to run.
```
sbatch t_run.sh
```


* Check the `slurm-<job-id>.slurm` file to see if there are any error.


* If no errors, Output should be saved in `./data/processed` directory after some time.

