import keras_tuner as kt
import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
import pandas as pd

from model import NBeatsNet
from loss import LOSSES
from data import M4Meta

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


def train_val_split(timeseries,targets,insample_size,outsample_size,window_sampling_limit,val_size=0.0):
    # print(f"training set has {x_train.shape[0]} samples")
    # x_test=last_insample_window(timeseries,insample_size)
    # y_test=targets
    # if val_size:
    val_size = int(val_size*len(timeseries))
    dataset = np.array(timeseries, dtype="object")
    np.random.shuffle(dataset)
    val,training = dataset[:val_size], dataset[val_size:]
    x_train,y_train=windowed_time_series(training, insample_size, outsample_size, window_sampling_limit)
    x_val,y_val=windowed_time_series(val, insample_size, outsample_size, window_sampling_limit)

    return x_train,y_train,x_val,y_val

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

    def __init__(self,freq,**kwargs):
        super(HyperNBeats, self).__init__(**kwargs)
        self.horizon=M4Meta.horizons_map[freq]
        # self.l_h=M4Meta.history_size[freq]
        self.steps=M4Meta.iterations[freq]
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
        model_type=hp.Choice("model_type",["interpretable","generic"])
        if model_type=="generic":
            stacks_num=hp.Int("stacks_num", 10, 30)
            stacks=[NBeatsNet.GENERIC_BLOCK]*stacks_num
            thetas_dim=[32]*stacks_num
            hidden_layer_units=[512]*stacks_num
            share_weights_in_stack=False
            nb_blocks_per_stack=1
        else:
            stacks=(NBeatsNet.TREND_BLOCK, NBeatsNet.SEASONALITY_BLOCK)
            thetas_dim=(hp.Int("degree_of_polynomial",min_value=2,max_value=4)+1,4)
            hidden_layer_units=(
                int(2**hp.Int("trend_layer_units_power",min_value=7,max_value=11,step=1)),
                int(2**hp.Int("season_layer_units_power",min_value=8,max_value=11,step=1)),
            )
            share_weights_in_stack=True
            nb_blocks_per_stack=hp.Int("nb_blocks_per_stack",min_value=3,max_value=5)
        # print(hidden_layer_units)

        def _create_model():
            net = NBeatsNet(
                # stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
                stack_types=stacks,
                nb_blocks_per_stack=nb_blocks_per_stack,
                forecast_length=self.outsample_size,
                backcast_length=self.insample_size,
                hidden_layer_units=hidden_layer_units,
                thetas_dim=thetas_dim,
                share_weights_in_stack=share_weights_in_stack,
                nb_harmonics=hp.Int("nb_harmonics",min_value=0,max_value=2),
                use_mase=self.use_mase,
                mase_frequency=self.frequency
            )
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                    hp.Float("init_lr",min_value=1e-5,max_value=1e-3,sampling="log"),
                    decay_steps=self.steps // 3,
                    decay_rate=hp.Float("lr_decay_rate",min_value=0.3,max_value=0.8,sampling="linear"),
                    staircase=True),
                clipnorm=1.0,
                epsilon=1e-8,
                # clipvalue=0.5
            )
            net.compile(loss=LOSSES[loss_fn], optimizer=optimizer)

            return net

        if hp.Boolean("parallel",False):
            strategy = tf.distribute.MirroredStrategy()
            print("Number of devices: {}".format(strategy.num_replicas_in_sync))
            with strategy.scope():
                net=_create_model()
        else:
            net=_create_model()
        return net.models['forecast']

    def dataset_for_training(self,hp,x,y):
        history_size_in_horizons=hp.Float("history_size",min_value=1.5,max_value=10.0,sampling="linear")
        # history_size_in_horizons=self.l_h
        window_sampling_limit=int(history_size_in_horizons*self.horizon)
        batch_size=int(2**hp.Int("batch_size_power",min_value=9,max_value=12))
        val_size=hp.Float("val_size", 0.0, 1.0,sampling="linear")
        x_train,y_train,x_test,y_test=train_val_split(x, y, self.insample_size, self.outsample_size, window_sampling_limit,val_size)

        train_ds=generate_dataset(x_train, y_train,batch_size,shuffle=(not self.use_mase),use_mase=self.use_mase)
        test_ds=generate_dataset(x_test, y_test,batch_size,use_mase=self.use_mase)
        return train_ds,test_ds

    def fit(self, hp,model,x,y,**kwargs):

        train_ds,test_ds=self.dataset_for_training(hp,x,y)

        return model.fit(
            x=train_ds,
            validation_data=test_ds,
            epochs=hp.Int("epochs",20,300),
            **kwargs
        )
