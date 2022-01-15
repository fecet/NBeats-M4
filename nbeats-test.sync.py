# %%

from functools import partial
from itertools import product
import os
from pathlib import Path
import warnings
import fire
import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from hypernbeat import HyperNBeats,M4Meta
from loss import LOSSES

# %%

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
else:
    from tqdm import tqdm
    os.environ["CUDA_VISIBLE_DEVICES"]="1" 
# %%

# %%

tf.config.threading.set_inter_op_parallelism_threads(8)
DATA_PATH=Path("./data/Dataset")
warnings.filterwarnings("ignore")

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
        # outsample_size=M4Meta.horizons_map[freq]
        x_test=last_insample_window(timeseries,insample_size)
        # y_test=targets
        y_pred=model.predict((x_test,targets)).squeeze()
    else:
        insample_size=model.input_shape[1]
        # outsample_size=M4Meta.horizons_map[freq]
        x_test=last_insample_window(timeseries,insample_size)
        # y_test=targets
        y_pred=model.predict(x_test).squeeze()
    # preds.append(y_pred)
    # score=LOSSES['smape'](targets, y_pred)
    # if score<20.0:
    return y_pred

def read_data(freq):
    filename_train = DATA_PATH/f'Train/{freq}-train.csv'
    filename_test  = DATA_PATH/f'Test/{freq}-test.csv'
    df=pd.read_csv(filename_train)
    tss=df.drop('V1',axis=1).values.copy(order='C').astype(np.float32)
    def dropna(x):
        return x[~np.isnan(x)]

    timeseries=[dropna(ts) for ts in tss]
    df=pd.read_csv(filename_test)
    targets=df.drop('V1',axis=1).values.copy(order='C').astype(np.float32)
    return timeseries,targets

# %%

def ensemble_member(freq,lookback,loss,overwrite=False):

    timeseries,targets=read_data(freq)

    history_size=M4Meta.history_size[freq]
    batch_size_power=10

    project_name=f"{freq}_{lookback}_{loss}"

    print(project_name)

    hp=kt.HyperParameters()
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

    tuner = kt.RandomSearch(
        HyperNBeats(freq),
        hyperparameters=hp,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=40,
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

    # sorted_preds=sorted(preds,key=lambda x:smape_loss(targets, x))
    evaluate_path=Path('./nbeats_result')/project_name
    evaluate_path.mkdir(exist_ok=True,parents=True)

    # score=tuner.oracle.get_best_trials(1)[0].score
    # print(f"{project_name}:{score}")

    for i,model in tqdm(enumerate(tuner.get_best_models(num_models=40))):
        y_pred=predict_m4_timeseries(timeseries, targets, model, loss) 
        np.save(evaluate_path/f"{i}.npy",y_pred)

    # print(tuner.results_summary())

    return tuner

# %%

def evaluate(results,freq="Yearly"):
    timeseries,targets=read_data(freq)
    preds=[np.load(fp) for fp in results]
    print(f"Ensembling from {len(preds)} models:")
    smape_loss=LOSSES['smape']
    print(f"Median ensemble: {smape_loss(targets, np.median(np.stack(preds),axis=0))}")
    print(f"Mean ensemble: {smape_loss(targets, np.mean(np.stack(preds),axis=0))}")


# %%

if __name__ == "__main__":

    # freq="Yearly"
    # results=Path('./nbeats_result').glob(f"{freq}_[234567]_*ma?e/[0-9].npy")
    # evaluate(results,freq)

    for freq,lookback,loss in product(["Yearly"],[2,3,4,5,6,7],LOSSES.keys()):
        ensemble_member(freq,lookback,loss)
