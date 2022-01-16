import tensorflow as tf

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

