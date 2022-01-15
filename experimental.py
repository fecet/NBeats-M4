from hypernbeat import HyperNBeats
import keras_tuner as kt
# %%
tf.config.threading.set_inter_op_parallelism_threads(8)
DATA_PATH=Path("./data/Dataset")


# %%


info_df=pd.read_csv(DATA_PATH/'M4-info.csv')

@dataclass()
class M4Meta:
    ids=info_df.M4id.values
    groups=info_df.SP.values
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }
    history_size = {
    'Yearly': 1.5,
    'Quarterly': 1.5,
    'Monthly': 1.5,
    'Weekly': 10,
    'Daily': 10,
    'Hourly': 10
}
    iterations = {
    'Yearly': 15000,
    'Quarterly': 15000,
    'Monthly': 15000,
    'Weekly': 5000,
    'Daily': 5000,
    'Hourly': 5000
}


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

def predict_m4_timeseries(timeseries, target, model, loss):
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


# %%


def mape_loss(y_true,y_pred):
    """
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    # mask=tf.cast(~tnp.isnan(y_true),tf.float32)
    condition=tf.cast(y_true,tf.bool)
    weights=tf.where(condition,1./y_true,.0)
    # weights = 1/y_true*mask
    # return 200 * tnp.nanmean(tf.abs(y_pred - y_true)*weights )


def smape_loss(y_true,y_pred):
    """
    sMAPE loss as defined in "Appendix A" of
    http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf
    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    # mask=tf.where(y_true,1.,0.)
    mask=tf.cast(y_true,tf.bool)
    mask=tf.cast(mask,tf.float32)
    sym_sum= tf.abs(y_true)+tf.abs(y_pred) 
    condition=tf.cast(sym_sum,tf.bool)
    weights=tf.where(condition,1./( sym_sum + 1e-8),0.0)
    # weights=tf.stop_gradient(weights)
    res=tf.abs(y_pred - y_true)*weights * mask
    nonzero=tf.math.count_nonzero(res)
    nonzero=tf.cast(nonzero,tf.float32)     

    return 200 * tf.math.reduce_sum(res)/nonzero

LOSSES={
    'mase':None, # implement inside model
    'mape':mape_loss,
    'smape':smape_loss,
}


def read_data(freq):
    pwd=Path('.')
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


def experimental_once(freq,lookback,loss,overwrite=False):

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
    (evaluate_path).mkdir(exist_ok=True,parents=True)

    # score=tuner.oracle.get_best_trials(1)[0].score
    # print(f"{project_name}:{score}")

    for i,model in tqdm(enumerate(tuner.get_best_models(num_models=20))):
        y_pred=predict_m4_timeseries(timeseries, targets, model, loss) 
        np.save(evaluate_path/f"{i}.npy",y_pred)

    # print(tuner.results_summary())

    return tuner
