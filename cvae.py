from keras.layers import Input, Dense
from keras.layers import BatchNormalization, Dropout, Flatten, Reshape, Lambda
from keras.layers import concatenate
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, MaxPooling2D, Dense, UpSampling2D, Reshape, Flatten, BatchNormalization, Lambda
from keras.layers import LeakyReLU, Input, Dropout, Conv2DTranspose
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras import losses
from keras.models import load_model


def create_cvae(input_shape, num_classes, latent_dim, dropout_rate, batch_size,
                start_lr=0.001):
    models = {}

    def apply_bn_and_dropout(x):
        return Dropout(dropout_rate)(BatchNormalization()(x))

    input_img = Input(shape=(input_shape[0], input_shape[1], 1))
    flatten_img = Flatten()(input_img)
    input_lbl = Input(shape=(num_classes,), dtype='float32')

    x = concatenate([flatten_img, input_lbl])
    x = Dense(1024, activation='relu')(x)
    x = Reshape((32, 32, 1))(x)
    x = apply_bn_and_dropout(x)
    x = Conv2D(16, (8, 8), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = apply_bn_and_dropout(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = apply_bn_and_dropout(x)
    x = Flatten()(x)

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    models["encoder"] = Model([input_img, input_lbl], l, 'Encoder')
    models["z_meaner"] = Model([input_img, input_lbl], z_mean, 'Enc_z_mean')
    models["z_lvarer"] = Model([input_img, input_lbl], z_log_var,
                               'Enc_z_log_var')

    z = Input(shape=(latent_dim,))
    input_lbl_d = Input(shape=(num_classes,), dtype='float32')

    x = concatenate([z, input_lbl_d])
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    #x = apply_bn_and_dropout(x)
    x = Reshape((16, 16, 1))(x)
    x = apply_bn_and_dropout(x)
    x = Conv2DTranspose(32, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = apply_bn_and_dropout(x)
    x = Conv2DTranspose(16, (8, 8), activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = apply_bn_and_dropout(x)
    x = Conv2DTranspose(1, (16, 16), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(input_shape[0] * input_shape[1], activation='sigmoid')(x)
    decoded = Reshape((input_shape[0], input_shape[1], 1))(x)

    models["decoder"] = Model([z, input_lbl_d], decoded, name='Decoder')
    models["cvae"] = Model([input_img, input_lbl, input_lbl_d],
                           models["decoder"](
                               [models["encoder"]([input_img, input_lbl]),
                                input_lbl_d]),
                           name="CVAE")
    models["style_t"] = Model([input_img, input_lbl, input_lbl_d],
                              models["decoder"](
                                  [models["z_meaner"]([input_img, input_lbl]),
                                   input_lbl_d]),
                              name="style_transfer")

    def vae_loss(x, decoded):
        x = K.reshape(x, shape=(batch_size, input_shape[0] * input_shape[1]))
        decoded = K.reshape(decoded,
                            shape=(batch_size, input_shape[0] * input_shape[1]))
        xent_loss = input_shape[0] * input_shape[1] * \
                    binary_crossentropy(x, decoded)
        kl_loss = -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
        )
        return (xent_loss + kl_loss) / 2 / input_shape[0] / input_shape[1]

    models["cvae"].compile(optimizer=Adam(start_lr), loss=vae_loss)

    return models, vae_loss


def create_simple_cvae(input_shape, num_classes, latent_dim, dropout_rate, batch_size,
                start_lr=0.001):
    models = {}

    def apply_bn_and_dropout(x):
        return Dropout(dropout_rate)(BatchNormalization()(x))

    input_img = Input(shape=(input_shape[0], input_shape[1], 1))
    flatten_img = Flatten()(input_img)
    input_lbl = Input(shape=(num_classes,), dtype='float32')

    x = concatenate([flatten_img, input_lbl])
    x = Dense(256, activation='relu')(x)
    x = apply_bn_and_dropout(x)

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    models["encoder"] = Model([input_img, input_lbl], l, 'Encoder')
    models["z_meaner"] = Model([input_img, input_lbl], z_mean, 'Enc_z_mean')
    models["z_lvarer"] = Model([input_img, input_lbl], z_log_var,
                               'Enc_z_log_var')

    z = Input(shape=(latent_dim,))
    input_lbl_d = Input(shape=(num_classes,), dtype='float32')

    x = concatenate([z, input_lbl_d])
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(input_shape[0] * input_shape[1], activation='sigmoid')(x)
    decoded = Reshape((input_shape[0], input_shape[1], 1))(x)

    models["decoder"] = Model([z, input_lbl_d], decoded, name='Decoder')
    models["cvae"] = Model([input_img, input_lbl, input_lbl_d],
                           models["decoder"](
                               [models["encoder"]([input_img, input_lbl]),
                                input_lbl_d]),
                           name="CVAE")
    models["style_t"] = Model([input_img, input_lbl, input_lbl_d],
                              models["decoder"](
                                  [models["z_meaner"]([input_img, input_lbl]),
                                   input_lbl_d]),
                              name="style_transfer")

    def vae_loss(x, decoded):
        x = K.reshape(x, shape=(batch_size, input_shape[0] * input_shape[1]))
        decoded = K.reshape(decoded,
                            shape=(batch_size, input_shape[0] * input_shape[1]))
        xent_loss = input_shape[0] * input_shape[1] * \
                    binary_crossentropy(x, decoded)
        kl_loss = -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
        )
        return (xent_loss + kl_loss) / 2 / input_shape[0] / input_shape[1]

    models["cvae"].compile(optimizer=Adam(start_lr), loss=vae_loss)

    return models, vae_loss

