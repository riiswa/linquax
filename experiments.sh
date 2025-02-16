rm -rf results
export JAX_ENABLE_X64=True
python run_experiment.py --multirun hydra/launcher=joblib 'seed=range(16)' policy=OFULQ,TS,MED env_id=inverted_pendulum
python run_experiment.py --multirun hydra/launcher=joblib 'seed=range(16)' policy=OFULQ,TS,MED env_id=uav
python run_experiment.py --multirun hydra/launcher=joblib 'seed=range(16)' policy=OFULQ,TS,MED env_id=boeing747
python make_plot.py