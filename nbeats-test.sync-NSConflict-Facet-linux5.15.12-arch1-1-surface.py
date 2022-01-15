# %%
# %load_ext tensorboard
# %tensorboard --logdir logs --port 8890
# from lets_plot import *
# LetsPlot.setup_html()
# from loguru import logger
# logger.add("file_log.log",level="ERROR")
# %%

from collections import OrderedDict
import csv
from dataclasses import dataclass
from functools import partial
from itertools import islice, product
import os
from pathlib import Path
import warnings

import fire
import keract
import keras_tuner as kt
from lets_plot import *
import numpy as np
import pandas as pd
from rich import inspect
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add, Dense, Input, Lambda, Reshape, Subtract
from tensorflow.keras.models import Model
import tensorflow_io as tfio
from tqdm.notebook import tqdm
# %%

tf.config.threading.set_inter_op_parallelism_threads(8)

DATA_PATH=Path("./data/Dataset")

# %%

def mase_loss(y_true,y_pred,input,frequency):
    """
    MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf
    :param forecast: Forecast values. Shape: batch, time_o
    :param insample: Insample values. Shape: batch, time_i
    :param outsample: Target values. Shape: batch, time_o
    :param frequency: Frequency value
    :return: Same shape array with error calculated for each time step
    """
    mask=tf.cast(y_true,tf.bool)
    mask=tf.cast(mask,tf.float32)
    
    seas_diff=tf.abs(input[:-frequency] - input[frequency:])
    scale =tf.reduce_mean(seas_diff)

    return tf.reduce_mean(tf.abs(y_true - y_pred)/scale*mask)
# tf.config.threading.get_inter_op_parallelism_threads()
# from nbeats_keras.model import NBeatsNet
# %%
class NBeatsNet:
    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'

    _BACKCAST = 'backcast'
    _FORECAST = 'forecast'

    def __init__(self,
                 input_dim=1,
                 exo_dim=0,
                 backcast_length=10,
                 forecast_length=2,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=(256,256),
                 nb_harmonics=None,
                 use_mase=False,
                 mase_frequency=0,
    ):

        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.input_dim = input_dim
        self.exo_dim = exo_dim
        self.input_shape = (self.backcast_length, self.input_dim)
        self.exo_shape = (self.backcast_length, self.exo_dim)
        self.output_shape = (self.forecast_length, self.input_dim)
        self.weights = {}
        self.nb_harmonics = nb_harmonics

        self.use_mase=use_mase
        self.mase_frequency=mase_frequency

        assert len(self.stack_types) == len(self.thetas_dim)

        x = Input(shape=self.input_shape, name='input_variable')
        x_ = {}
        for k in range(self.input_dim):
            x_[k] = Lambda(lambda z: z[..., k])(x)
        e_ = {}
        if self.has_exog():
            e = Input(shape=self.exo_shape, name='exos_variables')
            for k in range(self.exo_dim):
                e_[k] = Lambda(lambda z: z[..., k])(e)
        else:
            e = None
        y_ = {}

        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly = int(self.thetas_dim[stack_id])
            layer_size = int(self.units[stack_id])
            for block_id in range(self.nb_blocks_per_stack):
                backcast, forecast = self.create_block(x_, e_, stack_id, block_id, stack_type, nb_poly,layer_size)
                for k in range(self.input_dim):
                    x_[k] = Subtract()([x_[k], backcast[k]])
                    if stack_id == 0 and block_id == 0:
                        y_[k] = forecast[k]
                    else:
                        y_[k] = Add()([y_[k], forecast[k]])

        for k in range(self.input_dim):
            y_[k] = Reshape(target_shape=(self.forecast_length, 1))(y_[k])
            x_[k] = Reshape(target_shape=(self.backcast_length, 1))(x_[k])
        if self.input_dim > 1:
            y_ = Concatenate()([y_[ll] for ll in range(self.input_dim)])
            x_ = Concatenate()([x_[ll] for ll in range(self.input_dim)])
        else:
            y_ = y_[0]
            x_ = x_[0]

        if self.has_exog():
            n_beats_forecast = Model([x, e], y_, name=self._FORECAST)
            n_beats_backcast = Model([x, e], x_, name=self._BACKCAST)
        else:
            n_beats_forecast = Model(x, y_, name=self._FORECAST)
            n_beats_backcast = Model(x, x_, name=self._BACKCAST)
        if self.use_mase:
            y = Input(shape=self.output_shape, name='target_variable')
            n_beats_forecast = Model([x,y], y_, name=self._FORECAST)
            n_beats_forecast.add_loss(mase_loss(y,y_,x,self.mase_frequency))

        self.models = {model.name: model for model in [n_beats_backcast, n_beats_forecast]}
        self.cast_type = self._FORECAST

    def has_exog(self):
        # exo/exog is short for 'exogenous variable', i.e. any input
        # features other than the target time-series itself.
        return self.exo_dim > 0

    @staticmethod
    def load(filepath, custom_objects=None, compile=True):
        from tensorflow.keras.models import load_model
        return load_model(filepath, custom_objects, compile)

    def _r(self, layer_with_weights, stack_id):
        # mechanism to restore weights when block share the same weights.
        # only useful when share_weights_in_stack=True.
        if self.share_weights_in_stack:
            layer_name = layer_with_weights.name.split('/')[-1]
            try:
                reused_weights = self.weights[stack_id][layer_name]
                return reused_weights
            except KeyError:
                pass
            if stack_id not in self.weights:
                self.weights[stack_id] = {}
            self.weights[stack_id][layer_name] = layer_with_weights
        return layer_with_weights

    def create_block(self, x, e, stack_id, block_id, stack_type, nb_poly,units):
        # register weights (useful when share_weights_in_stack=True)
        def reg(layer):
            return self._r(layer, stack_id)

        # update name (useful when share_weights_in_stack=True)
        def n(layer_name):
            return '/'.join([str(stack_id), str(block_id), stack_type, layer_name])

        backcast_ = {}
        forecast_ = {}
        d1 = reg(Dense(units, activation='relu', name=n('d1')))
        d2 = reg(Dense(units, activation='relu', name=n('d2')))
        d3 = reg(Dense(units, activation='relu', name=n('d3')))
        d4 = reg(Dense(units, activation='relu', name=n('d4')))
        if stack_type == 'generic':
            theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = reg(Dense(self.backcast_length, activation='linear', name=n('backcast')))
            forecast = reg(Dense(self.forecast_length, activation='linear', name=n('forecast')))
        elif stack_type == 'trend':
            theta_f = theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f_b')))
            backcast = Lambda(trend_model, arguments={'is_forecast': False, 
                                                      'backcast_length': self.backcast_length,
                                                      'forecast_length': self.forecast_length,
            })
            forecast = Lambda(trend_model, arguments={'is_forecast': True, 
                                                      'backcast_length': self.backcast_length,
                                                      'forecast_length': self.forecast_length})
        else:  # 'seasonality'
            if self.nb_harmonics:
                theta_size=4*int(self.nb_harmonics/2*self.forecast_length-self.nb_harmonics+1)
                theta_b = reg(Dense(theta_size, activation='linear', use_bias=False, name=n('theta_b')))
            else:
                theta_b = reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = Lambda(seasonality_model,
                              arguments={'is_forecast': False, 
                                         'backcast_length': self.backcast_length,
                                         'forecast_length': self.forecast_length})
            forecast = Lambda(seasonality_model,
                              arguments={'is_forecast': True, 
                                         'backcast_length': self.backcast_length,
                                         'forecast_length': self.forecast_length})
        for k in range(self.input_dim):
            if self.has_exog():
                d0 = Concatenate()([x[k]] + [e[ll] for ll in range(self.exo_dim)])
            else:
                d0 = x[k]
            d1_ = d1(d0)
            d2_ = d2(d1_)
            d3_ = d3(d2_)
            d4_ = d4(d3_)
            theta_f_ = theta_f(d4_)
            theta_b_ = theta_b(d4_)
            backcast_[k] = backcast(theta_b_)
            forecast_[k] = forecast(theta_f_)

        return backcast_, forecast_

    def __getattr__(self, name):
        # https://github.com/faif/python-patterns
        # model.predict() instead of model.n_beats.predict()
        # same for fit(), train_on_batch()...
        attr = getattr(self.models[self._FORECAST], name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            cast_type = self._FORECAST
            if attr.__name__ == 'predict' and 'return_backcast' in kwargs and kwargs['return_backcast']:
                del kwargs['return_backcast']
                cast_type = self._BACKCAST
            return getattr(self.models[cast_type], attr.__name__)(*args, **kwargs)

        return wrapper

def linear_space(backcast_length, forecast_length, is_forecast=True):
    # ls = K.arange(-float(backcast_length), float(forecast_length), 1) / forecast_length
    # return ls[backcast_length:] if is_forecast else K.abs(K.reverse(ls[:backcast_length], axes=0))
    horizon = forecast_length if is_forecast else backcast_length
    return K.arange(0,horizon)/horizon


def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)])
    s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)])
    if p == 1:
        s = s2
    else:
        s = K.concatenate([s1, s2], axis=0)
    s = K.cast(s, np.float32)
    return K.dot(thetas, s)


def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    t = K.stack([t ** i for i in range(p)]) # p*backcast 
    t = K.cast(t, np.float32)
    return K.dot(thetas, t) #batch size * backcast
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

def plot_from_history(history):
    data={}
    val_data={}
    for key in history.history.keys():
        if key.startswith('val'):
            val_data[key]=history.history[key]
            val_data[f'_{key}']=[key]*len(val_data[key])
        else:
            data[key]=history.history[key]
            data[f'_{key}']=[key]*len(data[key])
    data['epoch']=np.array(history.epoch)+1
    epochs=len(data['epoch'])
    val_epochs=len(val_data['val_loss'])
    val_freq=epochs // val_epochs
    val_data['epoch']=np.arange(val_freq,epochs,val_freq)
    plot=(
        ggplot(data,aes(x='epoch'))+
        geom_line(aes(y='loss',color='_loss'))+
        geom_line(aes(y='val_loss',color='_val_loss'),data=val_data)+
        labs(color="Legend text")+
        ylim(0,0.3)
    )

    return plot

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
def smape_loss_legacy(y_true,y_pred):
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
    weights=tf.stop_gradient(weights)
    return 200 * tnp.nanmean(tf.abs(y_pred - y_true)*weights * mask)
    # return 200 * tnp.nanmean(tf.abs(y_pred - y_true)*weights )
# def mase_loss(insample: t.Tensor, freq: int,
#               forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
#     """
#     MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf
#     :param insample: Insample values. Shape: batch, time_i
#     :param freq: Frequency value
#     :param forecast: Forecast values. Shape: batch, time_o
#     :param target: Target values. Shape: batch, time_o
#     :param mask: 0/1 mask. Shape: batch, time_o
#     :return: Loss value
#     """
#     masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
#     masked_masep_inv = divide_no_nan(mask, masep[:, None])
#     return t.mean(t.abs(target - forecast) * masked_masep_inv)


LOSSES={
    'mape':mape_loss,
    'smape':smape_loss,
    'smape2':smape_loss_legacy
}


def group_values(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
    """
    Filter values array by group indices and clean it from NaNs.
    :param values: Values to filter.
    :param groups: Timeseries groups.
    :param group_name: Group name to filter by.
    :return: Filtered and cleaned timeseries.
    """
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])
def summarize_groups(scores,others=True):
    """
    Re-group scores respecting M4 rules.
    :param scores: Scores per group.
    :return: Grouped scores.
    """
    scores_summary = OrderedDict()

    def group_count(group_name):
        return len(np.where(M4Meta.groups == group_name)[0])
    weighted_score = {}
    for g in ['Yearly', 'Quarterly', 'Monthly']:
        weighted_score[g] = scores[g] * group_count(g)
        scores_summary[g] = scores[g]

    for g in ['Weekly', 'Daily', 'Hourly']:
        weighted_score[g] = scores[g] * group_count(g)
        scores_summary[g] = scores[g]
    if others:
        others_score = 0
        others_count = 0
        for g in ['Weekly', 'Daily', 'Hourly']:
            others_score += scores[g] * group_count(g)
            others_count += group_count(g)
        weighted_score['Others'] = others_score
        scores_summary['Others'] = others_score / others_count

    average = np.sum(list(weighted_score.values())) / len(M4Meta.groups)
    scores_summary['Average'] = average

    return scores_summary

def evaluate_result(forecast,targets,others=True):
    forecast = np.array([v[~np.isnan(v)] for v in forecast],dtype='object')
    results={
    freq:smape_loss(
        group_values(targets,M4Meta.groups,freq),
        group_values(forecast,M4Meta.groups,freq)
    ).numpy() for freq in M4Meta.seasonal_patterns
    }
    return pd.DataFrame(summarize_groups(results),index=['smape'])


# %%

def read_data(freq):
    filename_train = DATA_PATH/f'Train/{freq}-train.csv'
    filename_test = DATA_PATH/f'Test/{freq}-test.csv'
    df=pd.read_csv(filename_train)
    tss=df.drop('V1',axis=1).values.copy(order='C').astype(np.float32)
    def dropna(x):
        return x[~np.isnan(x)]

    timeseries=[dropna(ts) for ts in tss]
    df=pd.read_csv(filename_test)
    targets=df.drop('V1',axis=1).values.copy(order='C').astype(np.float32)
    return timeseries,targets

def windowed_time_series(timeseries,insample_size,outsample_size,window_sampling_limit):
    insamples=[]
    outsamples=[]
    for sampled_timeseries in timeseries:
        gen_size=min(window_sampling_limit,len(sampled_timeseries)-1)
        insample = np.zeros((gen_size,insample_size),dtype=np.float32)
        outsample = np.zeros((gen_size,outsample_size),dtype=np.float32)
        for idx,cut_point in enumerate(np.arange(
                        start=max(1, len(sampled_timeseries) - window_sampling_limit),
                        stop=len(sampled_timeseries),
                        dtype=int)):
            insample_window = sampled_timeseries[max(0, cut_point - insample_size):cut_point]
            insample[idx,-len(insample_window):] = insample_window
            outsample_window = sampled_timeseries[
                               cut_point:min(len(sampled_timeseries), cut_point + outsample_size)]
            outsample[idx,:len(outsample_window)] = outsample_window
        insamples.append(insample)
        outsamples.append(outsample)
    x_train=np.concatenate(insamples)
    y_train=np.concatenate(outsamples)  
    print(x_train.shape,y_train.shape)
    return x_train, y_train


def timeseries_sampler(timeseries,targets,insample_size,outsample_size,window_sampling_limit,val_size=0.0):
    # print(f"training set has {x_train.shape[0]} samples")
    x_test=last_insample_window(timeseries,insample_size)
    y_test=targets
    if val_size:
        val_size = int(val_size*len(timeseries))
        dataset = np.array(timeseries, dtype="object")
        np.random.shuffle(dataset)
        val,training = dataset[:val_size], dataset[val_size:]
        x_train,y_train=windowed_time_series(training, insample_size, outsample_size, window_sampling_limit)
        x_val,y_val=windowed_time_series(val, insample_size, outsample_size, window_sampling_limit)
        return x_train,y_train,x_val,y_val,x_test,y_test
    else:
        x_train,y_train=windowed_time_series(timeseries, insample_size, outsample_size, window_sampling_limit)

    return x_train,y_train,x_test,y_test

def generate_dataset(x_train,y_train,batch_size=1024,shuffle=False,use_mase=False):
    batch_size=int(batch_size)
    shuffle_buffer=len(x_train)
    ds=tfio.experimental.IODataset.from_numpy(( x_train,y_train ))
    if use_mase:
        label_ds=tfio.experimental.IODataset.from_numpy(y_train)
        ds=tf.data.Dataset.zip((ds,label_ds))
    if shuffle:
        ds=ds.shuffle(shuffle_buffer,reshuffle_each_iteration=True)
    return ds.batch(batch_size)


class HyperNBeats(kt.HyperModel):

    def __init__(self,freq,repeat=False,**kwargs):
        super(HyperNBeats, self).__init__(**kwargs)
        self.horizon=M4Meta.horizons_map[freq]
        # self.l_h=M4Meta.history_size[freq]
        self.steps=M4Meta.iterations[freq]
        self.repeat=repeat
        self.frequency=M4Meta.frequency_map[freq]
        # self.val_size=val_size

    def build(self, hp:kt.HyperParameters)->tf.keras.Model:
        """Build Hyper NBeatsNet

        """
        hp.Int("dummy", 0, 100)
        lookback=hp.Int("lookback",2,7)
        self.insample_size=lookback*self.horizon
        self.outsample_size=self.horizon
        loss_fn=hp.Choice("loss_fn",["smape","mape","mase"])
        self.use_mase=(loss_fn=="mase")
        net = NBeatsNet(
            # stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
            stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK),
            nb_blocks_per_stack=hp.Int("nb_blocks_per_stack",min_value=3,max_value=5),
            forecast_length=self.outsample_size,
            backcast_length=self.insample_size,
            hidden_layer_units=(
                2**hp.Int("trend_layer_units_power",min_value=7,max_value=11,step=1),
                2**hp.Int("season_layer_units_power",min_value=8,max_value=11,step=1),
            ),
            thetas_dim=(hp.Int("degree_of_polynomial",min_value=2,max_value=4)+1,4),
            share_weights_in_stack=True,
            nb_harmonics=hp.Int("nb_harmonics",min_value=0,max_value=2),
            use_mase=self.use_mase,
            mase_frequency=self.frequency
           )
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            hp.Float("init_lr",min_value=1e-5,max_value=1e-3,sampling="log"),
            # 1e-3,
            decay_steps=self.steps // 3,
            decay_rate=hp.Float("lr_decay_rate",min_value=0.3,max_value=0.8,sampling="linear"),
            # decay_rate=0.5,
            staircase=True)
        if self.use_mase:
            net.compile(loss=None, 
                # optimizer='adam',
                optimizer=tf.keras.optimizers.Adam(
                    # learning_rate=hp.Float("init_lr",min_value=1e-5,max_value=1e-4,sampling="linear"),
                    learning_rate=lr_schedule,
                    clipnorm=1.0,
                    epsilon=1e-8,
                    # clipvalue=0.5
                ),
            )
        else:
            net.compile(loss=LOSSES[loss_fn], 
                # optimizer='adam',
                optimizer=tf.keras.optimizers.Adam(
                    # learning_rate=hp.Float("init_lr",min_value=1e-5,max_value=1e-4,sampling="linear"),
                    learning_rate=lr_schedule,
                    clipnorm=1.0,
                    epsilon=1e-8,
                    # clipvalue=0.5
                ),
            )
        return net.models['forecast']

    def dataset_for_training(self,hp,x,y):
        history_size_in_horizons=hp.Float("history_size",min_value=1.5,max_value=10.0,sampling="linear")
        # history_size_in_horizons=self.l_h
        window_sampling_limit=int(history_size_in_horizons*self.horizon)
        batch_size=2**hp.Int("batch_size_power",min_value=9,max_value=12)
        val_size=hp.Float("val_size", 0.0, 1.0,sampling="linear")
        if val_size:
            print("use validation_data")
            x_train,y_train,x_test,y_test,_,_=timeseries_sampler(x, y, self.insample_size, self.outsample_size, window_sampling_limit,val_size)
        else:
            x_train,y_train,x_test,y_test=timeseries_sampler(x, y, self.insample_size, self.outsample_size, window_sampling_limit)

        train_ds=generate_dataset(x_train, y_train,batch_size,shuffle=(not self.use_mase),use_mase=self.use_mase)
        test_ds=generate_dataset(x_test, y_test,batch_size,use_mase=self.use_mase)
        if self.repeat:
            train_ds=train_ds.repeat()
        return train_ds,test_ds
    
    def fit(self, hp,model,x,y,**kwargs):

        # history_size_in_horizons=hp.Float("history_size",min_value=1.5,max_value=10.0,sampling="linear")
        # # history_size_in_horizons=self.l_h
        # window_sampling_limit=int(history_size_in_horizons*self.horizon)
        # x_train,y_train,x_test,y_test=timeseries_sampler(x, y, self.insample_size, self.outsample_size, window_sampling_limit)
        # train_ds,test_ds=generate_dataset(x_train, y_train, x_test, y_test,
        #     batch_size=2**hp.Int("batch_size_power",min_value=9,max_value=12),
        #     # batch_size=2**11
        # )
        train_ds,test_ds=self.dataset_for_training(hp,x,y)


        return model.fit(
            x=train_ds,
            validation_data=test_ds,
            epochs=hp.Int("epochs",20,300),
            **kwargs)

# %%


def experimental_once(freq,lookback,loss,history_size=None,overwrite=False,batch_size_power=10):

    timeseries,targets=read_data(freq)

    if history_size:
        project_name=f"{freq}_{lookback}_{loss}_{history_size}"
    else:
        history_size=M4Meta.history_size[freq]
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
        if loss=="mase":
            insample_size=model.input_shape[0][1]
            outsample_size=M4Meta.horizons_map[freq]
            x_test=last_insample_window(timeseries,insample_size)
            # y_test=targets
            y_pred=model.predict((x_test,targets)).squeeze()
        else:
            insample_size=model.input_shape[1]
            outsample_size=M4Meta.horizons_map[freq]
            x_test=last_insample_window(timeseries,insample_size)
            # y_test=targets
            y_pred=model.predict(x_test).squeeze()
        # preds.append(y_pred)
        score=LOSSES['smape'](targets, y_pred)
        if score<20.0:
            np.save(evaluate_path/f"{i}.npy",y_pred)

    # print(tuner.results_summary())

    return tuner

def experimental(freqs,lookbacks,losses):
    for freq,lookback,loss in product(
        freqs,lookbacks,losses
    ):
        print((freq,lookback,loss))
    for freq,lookback,loss in product(
        freqs,lookbacks,losses
    ):
        experimental_once(freq, lookback, loss)

# %%

def results_in_small_dataset(freq="Yearly"):
    timeseries,targets=read_data(freq)
    # results=filter(lambda x: x.parent.name.startswith(freq),Path('./nbeats_result').glob(f"Yearly_?_mape*/*.npy"))
    # results=Path('./nbeats_result').glob(f"{freq}_[234567]_smape/[0123456789].npy")
    results=Path('./nbeats_result').glob(f"small_dataset/*.npy")
    # results=Path('./nbeats_result').glob(f"{freq}*/*.npy")
    preds=[np.load(fp) for fp in results]
    print(f"Ensembling from {len(preds)} models:")
    smape_loss=LOSSES['smape']
    print(f"Median ensemble: {smape_loss(targets, np.median(np.stack(preds),axis=0))}")
    print(f"Mean ensemble: {smape_loss(targets, np.mean(np.stack(preds),axis=0))}")


def results_in_large_dataset(freq="Yearly"):
    timeseries,targets=read_data(freq)
    # results=filter(lambda x: x.parent.name.startswith(freq),Path('./nbeats_result').glob(f"Yearly_?_mape*/*.npy"))
    # results=Path('./nbeats_result').glob(f"{freq}_[234567]_smape/[0123456789].npy")
    results=Path('./nbeats_result').glob(f"large_dataset/*.npy")
    # results=Path('./nbeats_result').glob(f"{freq}*/*.npy")
    preds=[np.load(fp) for fp in results]
    print(f"Ensembling from {len(preds)} models:")
    smape_loss=LOSSES['smape']
    print(f"Median ensemble: {smape_loss(targets, np.median(np.stack(preds),axis=0))}")
    print(f"Mean ensemble: {smape_loss(targets, np.mean(np.stack(preds),axis=0))}")

def results(freq="Yearly"):
    timeseries,targets=read_data(freq)
    # results=filter(lambda x: x.parent.name.startswith(freq),Path('./nbeats_result').glob(f"Yearly_?_mape*/*.npy"))
    # results=Path('./nbeats_result').glob(f"{freq}_[234567]_smape/[0123456789].npy")
    results=Path('./nbeats_result').glob(f"{freq}_[234567]_*ma?e/[0-9].npy")
    # results=Path('./nbeats_result').glob(f"{freq}*/*.npy")
    preds=[np.load(fp) for fp in results]
    print(f"Ensembling from {len(preds)} models:")
    smape_loss=LOSSES['smape']
    print(f"Median ensemble: {smape_loss(targets, np.median(np.stack(preds),axis=0))}")
    print(f"Mean ensemble: {smape_loss(targets, np.mean(np.stack(preds),axis=0))}")
# results()
# %%

def experimental_from_given_parameters(hp,freq):

    timeseries,targets=read_data(freq)
    hypermpdel=HyperNBeats(freq)
    es_callback=tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True,
        verbose=1
    )
    # logdir = Path("logs")/"keras_tuner_log"
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    model=hypermpdel.build(hp)
    train_ds,test_ds=hypermpdel.dataset_for_training(hp,timeseries,targets)

    model.fit(x=train_ds,
              validation_data=test_ds,
              epochs=5,
              # steps_per_epoch=M4Meta.iterations[freq] // 100,
              verbose=2,
              callbacks=[es_callback],
              # validation_freq=5,
              )

    insample_size=model.input_shape[0][1]
    outsample_size=M4Meta.horizons_map[freq]
    x_test=last_insample_window(timeseries,insample_size)
    # y_test=targets
    # y_pred=model.predict(x_test).squeeze()
    y_pred=model.predict((x_test,targets)).squeeze()
    # preds.append(y_pred)
    score=LOSSES['smape'](targets, y_pred)
    # sorted_preds=sorted(preds,key=lambda x:smape_loss(targets, x))
    return score,model

# %%

if __name__ == "__main__":
    fire.Fire(experimental_once)
    # for i in [2,3,4,5,6,7]:
    #     experimental_once("Yearly", i, "mase",overwrite=False)

# %%


# freq="Yearly"
# lookback=4
# loss="smape"
# batch_size_power=10
# history_size=1.5
# hp=kt.HyperParameters()
# hp.Fixed("lookback",lookback)
# hp.Fixed("nb_blocks_per_stack",3)
# hp.Fixed("trend_layer_units_power",8)
# hp.Fixed("season_layer_units_power",11)
# hp.Fixed("degree_of_polynomial",2)
# hp.Fixed("loss_fn",loss)
# hp.Fixed("batch_size_power",batch_size_power)
# hp.Fixed("init_lr",1e-3)
# hp.Fixed("history_size",history_size)
# hp.Fixed("lr_decay_rate", 0.5)
# hp.Fixed("nb_harmonics", 1)
# hp.Fixed("epochs", 100)
# hp.Fixed("val_size", 0.9)
# project_name="small_dataset"
# timeseries,targets=read_data(freq)
# tuner = kt.RandomSearch(
#     HyperNBeats(freq),
#     hyperparameters=hp,
#     objective=kt.Objective("val_loss", direction="min"),
#     max_trials=3,
#     overwrite=True,
#     directory="kt",
#     project_name=project_name,
#     seed=64
# )
# es_callback=tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', patience=20,
#     restore_best_weights=True,
#     verbose=1
# )

# logdir = Path("logs")/project_name
# tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# tuner.search(timeseries, targets, 
#             # epochs=100, 
#             # steps_per_epoch=M4Meta.iterations[freq] // 100,
#             verbose=2,
#             callbacks=[es_callback,tensorboard_callback]
#             )


# evaluate_path=Path('./nbeats_result')/project_name
# (evaluate_path).mkdir(exist_ok=True,parents=True)

# for i,model in tqdm(enumerate(tuner.get_best_models(num_models=20))):
#     insample_size=model.input_shape[1]
#     outsample_size=M4Meta.horizons_map[freq]
#     x_test=last_insample_window(timeseries,insample_size)
#     # y_test=targets
#     y_pred=model.predict(x_test).squeeze()
#     # preds.append(y_pred)
#     np.save(evaluate_path/f"{i}.npy",y_pred)

# %%

