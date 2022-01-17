# %%
import os
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  
if isnotebook():
    from tqdm.notebook import tqdm
    os.environ["CUDA_VISIBLE_DEVICES"]="0" 
    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" 
else:
    from tqdm import tqdm
    os.environ["CUDA_VISIBLE_DEVICES"]="1" 

# %%
from itertools import product
from pathlib import Path
import warnings
import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from collections.abc import Iterable
from hypernbeat import HyperNBeats
from loss import LOSSES
from data import M4Meta,read_data
tf.config.threading.set_inter_op_parallelism_threads(8)
warnings.filterwarnings("ignore")

# %%

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# %%

def last_insample_window(timeseries, insample_size):
    """
    The last window of insample size of all timeseries.
    This function does not support batching and does not reshuffle timeseries.
    :return: Last insample window of all timeseries. Shape "timeseries, insample size"
    """
    insample = np.zeros((len(timeseries), insample_size))
    for i, ts in enumerate(timeseries):
        ts_last_window = ts[-insample_size:]
        insample[i, -len(ts):] = ts_last_window
    return insample

def predict_m4_timeseries(timeseries, targets, model, loss):
    if loss=="mase":
        insample_size=model.input_shape[0][1]
        x_test=last_insample_window(timeseries,insample_size)
        y_pred=model.predict((x_test,targets)).squeeze()
    else:
        insample_size=model.input_shape[1]
        x_test=last_insample_window(timeseries,insample_size)
        y_pred=model.predict(x_test).squeeze()
    return y_pred


# %%

def ensemble_member(freq,lookback,loss,model_type="interpretable",overwrite=False,parallel=False):

    timeseries,targets=read_data(freq)

    history_size=M4Meta.history_size[freq]
    batch_size_power=14

    if model_type=="interpretable":
        project_name=f"{freq}_{lookback}_{loss}"
    else:
        project_name=f"{model_type}_{freq}_{lookback}_{loss}"
         

    print(project_name)

    hp=kt.HyperParameters()
    hp.Fixed("model_type", model_type)
    hp.Fixed("lookback",lookback)
    hp.Fixed("nb_blocks_per_stack",3)
    hp.Fixed("trend_layer_units_power",8)
    hp.Fixed("season_layer_units_power",11)
    hp.Fixed("degree_of_polynomial",2)
    hp.Fixed("loss_fn",loss)
    hp.Fixed("batch_size_power",batch_size_power)
    hp.Fixed("init_lr",1e-3)
    hp.Fixed("history_size",history_size)
    hp.Fixed("lr_decay_rate", 0.5)
    hp.Fixed("nb_harmonics", 1)
    hp.Fixed("epochs", 100)
    hp.Fixed("val_size", 0.1)
    hp.Fixed("stacks_num", 15)
    hp.Fixed("parallel", parallel)

    tuner = kt.RandomSearch(
        HyperNBeats(freq),
        hyperparameters=hp,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=15,
        overwrite=overwrite,
        directory="kt",
        project_name=project_name,
        seed=64
    )
    es_callback=tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20,
        restore_best_weights=True,
        verbose=1
    )

    logdir = Path("logs")/"keras_tuner_log"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    tuner.search(timeseries, targets, 
                # epochs=100, 
                # steps_per_epoch=M4Meta.iterations[freq] // 100,
                verbose=2,
                callbacks=[es_callback,tensorboard_callback]
                )

    evaluate_path=Path('./nbeats_result')/project_name
    evaluate_path.mkdir(exist_ok=True,parents=True)

    print("Recording pred...")
    for i,model in tqdm(enumerate(tuner.get_best_models(num_models=15))):
        y_pred=predict_m4_timeseries(timeseries, targets, model, loss) 
        np.save(evaluate_path/f"{i}.npy",y_pred)


    return tuner

# %%
class Evaluater(object):

    def __init__(self,freq):
        self._freq = freq
        self.timeseries,self.targets=read_data(freq)
        self.eval_fn=LOSSES['smape']

    def ensemble_results(self,results):
        if not isinstance(results, Iterable):
            results=[results]
        preds=[np.load(fp) for fp in results]
        print(f"Ensembling from {len(preds)} models:")
        # mean_ensemble=np.mean(np.stack(preds),axis=0)
        median_ensemble=np.median(np.stack(preds),axis=0)
        return median_ensemble

    def __call__(self,results):
        y_pred=self.ensemble_results(results)
        median_score=self.eval_fn(self.targets, y_pred).numpy()
        print(f"Median ensemble: {median_score}")
        return median_score

# %%

def generic_model(freq,repeat=10):
    results=[]
    for lookback,loss in product([2,3,4,5,6,7],LOSSES.keys()):
        project_name=f"generic_{freq}_{lookback}_{loss}"
        member_results=list(Path('./nbeats_result').glob(f"{project_name}/*.npy"))
        select_results=np.random.choice(member_results,size=repeat) # For accurate reporduction
        results.extend(select_results)
    return results

def interpretable_model(freq,repeat=10):
    results=[]
    for lookback,loss in product([2,3,4,5,6,7],LOSSES.keys()):
        project_name=f"{freq}_{lookback}_{loss}"
        member_results=list(Path('./nbeats_result').glob(f"{project_name}/*.npy"))
        select_results=np.random.choice(member_results,size=repeat) # For accurate reporduction
        results.extend(select_results)
    return results


# %%


# %%

if __name__ == "__main__":
    # freq="Yearly"
    # results=Path('./nbeats_result').glob(f"{freq}_[234567]_*ma?e/[0-9].npy")
    # evaluate(results,freq)
    for freq,lookback,loss in product(["Yearly"],[2,3,4,5,6,7],LOSSES.keys()):
        ensemble_member(freq,lookback,loss,model_type="generic")
        # ensemble_member(freq,lookback,loss)
#     evaluater=Evaluater(freq)
#     results=interpretable_model(freq)
# 
#     evaluater(results)
    

# %%


